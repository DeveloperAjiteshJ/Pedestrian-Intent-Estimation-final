// ============================================================================
// TinyMobileNet-XS FPGA Inference Core
// Top-Level Module for Nexys A7-100T
// ============================================================================

module tinymobilenet_top (
    input  wire       clk,
    input  wire       reset_button,
    input  wire       start_button,
    output reg        inference_done,
    input  wire       uart_rx,
    output wire       uart_tx,
    output reg  [6:0] seg,
    output reg  [7:0] an,
    output reg        dp
);

`ifdef SYNTHESIS
    localparam integer NUM_SAMPLES = 6;  // Synthesis: limited by BRAM capacity
`else
    localparam integer NUM_SAMPLES = 112;  // Simulation: full dataset
`endif
    localparam integer BYTES_PER_SAMPLE = 49152;
    localparam integer TOTAL_SAMPLE_BYTES = NUM_SAMPLES * BYTES_PER_SAMPLE;

    reg [3:0] state, next_state;
    wire rst_n_int;

    localparam IDLE        = 4'd0;
    localparam WAIT_BUTTON = 4'd1;
    localparam INFERENCE   = 4'd2;
    localparam OUTPUT      = 4'd3;
    localparam UART_TX     = 4'd4;

    reg [7:0] sample_rom [0:TOTAL_SAMPLE_BYTES-1];
    reg [7:0] label_rom [0:NUM_SAMPLES-1];
    wire [31:0] logits_0;
    wire [31:0] logits_1;
    wire inference_done_core;

    wire predicted_class;
    wire [7:0] confidence;

    reg [15:0] sample_count;
    reg [15:0] correct_count;
    reg [7:0] accuracy_pct;
    reg [6:0] sample_index;
    wire expected_class;
    wire [31:0] sample_base;
    reg [15:0] rom_ptr;

    reg [15:0] scan_div;
    reg [2:0] scan_sel;
    reg [3:0] digit_val;
    reg       start_button_d;
    wire      start_pulse;
    wire [3:0] conf_x;
    wire [3:0] conf_y;
    wire [3:0] conf_z;
    wire [3:0] conf_a;
    reg        displayed_pred_class;
    reg        displayed_expected_class;
    reg [7:0]  displayed_confidence;
    reg [3:0]  cycle_counter;  // Counter for display (0-10 for 11 samples, needs 4 bits)

    reg        infer_busy;
    reg [15:0] infer_ptr;
    reg        core_start;
    reg        core_valid;
    reg        core_last;
    reg [7:0]  core_byte;

    reg        uart_reset_pending;
    reg        uart_result_pending;
    reg        uart_packet_active;
    reg        uart_packet_is_result;
    reg [2:0]  uart_packet_index;
    reg        uart_start_tx;
    reg [6:0]  uart_result_sample_index;
    reg        uart_result_pred_class;
    reg        uart_result_expected_class;
    reg [7:0]  uart_result_confidence;
    wire       uart_busy;
    reg [7:0]  uart_frame_byte;

    localparam integer DIFF_BIAS = 34454;

    assign rst_n_int = ~reset_button;
    assign start_pulse = start_button & ~start_button_d;

    assign conf_x = ((confidence > 8'd99) ? 8'd99 : confidence) / 8'd10;
    assign conf_y = ((confidence > 8'd99) ? 8'd99 : confidence) % 8'd10;
    assign conf_z = (confidence[3:0] * 8'd10) / 8'd16;
    assign conf_a = ((confidence[3:0] * 8'd100) / 8'd16) % 8'd10;

    initial begin
`ifdef SYNTHESIS
        $readmemh("all_samples_synth.mem", sample_rom);
        $readmemb("sample_labels_synth.mem", label_rom);
`else
        $readmemh("all_samples.mem", sample_rom);
        $readmemb("sample_labels.mem", label_rom);
`endif
    end

    assign sample_base = sample_index * BYTES_PER_SAMPLE;
    assign expected_class = label_rom[sample_index][0];

    inference_core inference_inst (
        .clk(clk),
        .rst_n(rst_n_int),
        .start(core_start),
        .sample_byte(core_byte),
        .sample_valid(core_valid),
        .sample_last(core_last),
        .logits_0(logits_0),
        .logits_1(logits_1),
        .predicted_class(predicted_class),
        .confidence(confidence),
        .done(inference_done_core)
    );

    always @(posedge clk or negedge rst_n_int) begin
        if (!rst_n_int) begin
            state <= IDLE;
            start_button_d <= 1'b0;
        end else begin
            state <= next_state;
            start_button_d <= start_button;
        end
    end

    always @(*) begin
        next_state = state;
        case (state)
            IDLE:        next_state = WAIT_BUTTON;
            WAIT_BUTTON: if (start_pulse) next_state = INFERENCE;
            INFERENCE:   if (inference_done_core) next_state = OUTPUT;
            OUTPUT:      next_state = UART_TX;
            UART_TX:     next_state = WAIT_BUTTON;
            default:     next_state = IDLE;
        endcase
    end

    always @(posedge clk or negedge rst_n_int) begin
        if (!rst_n_int) begin
            rom_ptr <= 16'd0;
            infer_ptr <= 16'd0;
            core_start <= 1'b0;
            core_valid <= 1'b0;
            core_last <= 1'b0;
            core_byte <= 8'd0;
        end else begin
            if (state == INFERENCE) begin
                core_start <= (infer_ptr == 16'd0);
                if (infer_ptr < BYTES_PER_SAMPLE) begin
                    rom_ptr <= infer_ptr;
                    core_valid <= 1'b1;
                    core_byte <= sample_rom[sample_base + infer_ptr];
                    core_last <= (infer_ptr == (BYTES_PER_SAMPLE - 1));
                    infer_ptr <= infer_ptr + 16'd1;
                end else begin
                    core_valid <= 1'b0;
                    core_last <= 1'b0;
                end
            end else begin
                rom_ptr <= 16'd0;
                infer_ptr <= 16'd0;
                core_start <= 1'b0;
                core_valid <= 1'b0;
                core_last <= 1'b0;
            end
        end
    end

    always @(posedge clk or negedge rst_n_int) begin
        if (!rst_n_int) begin
            sample_count <= 16'd0;
            correct_count <= 16'd0;
            accuracy_pct <= 8'd0;
            sample_index <= 7'd0;
            inference_done <= 1'b0;
            displayed_pred_class <= 1'b0;
            displayed_expected_class <= 1'b0;
            displayed_confidence <= 8'd0;
            cycle_counter <= 4'd0;  // Initialize cycle counter (4 bits for 0-10)
        end else begin
            inference_done <= 1'b0;
            if (state == OUTPUT) begin
                displayed_pred_class <= predicted_class;
                displayed_expected_class <= expected_class;
                displayed_confidence <= confidence;
                sample_count <= sample_count + 16'd1;
                if (predicted_class == expected_class)
                    correct_count <= correct_count + 16'd1;

                if (sample_count != 16'd0)
                    accuracy_pct <= (((predicted_class == expected_class) ? (correct_count + 16'd1) : correct_count) * 8'd100) / (sample_count + 16'd1);
                else
                    accuracy_pct <= 8'd0;

                if (sample_index == (NUM_SAMPLES - 1)) begin
                    sample_index <= 7'd0;
                    cycle_counter <= 4'd0;  // Reset cycle counter after last sample
                end else begin
                    sample_index <= sample_index + 7'd1;
                    cycle_counter <= cycle_counter + 4'd1;  // Increment cycle counter
                end
                inference_done <= 1'b1;
            end
        end
    end

    always @(posedge clk or negedge rst_n_int) begin
        if (!rst_n_int) begin
            uart_reset_pending <= 1'b1;
            uart_result_pending <= 1'b0;
            uart_packet_active <= 1'b0;
            uart_packet_is_result <= 1'b0;
            uart_packet_index <= 3'd0;
            uart_start_tx <= 1'b0;
            uart_result_sample_index <= 7'd0;
            uart_result_pred_class <= 1'b0;
            uart_result_expected_class <= 1'b0;
            uart_result_confidence <= 8'd0;
        end else begin
            uart_start_tx <= 1'b0;

            if (state == OUTPUT) begin
                uart_result_pending <= 1'b1;
                uart_result_sample_index <= sample_index;
                uart_result_pred_class <= predicted_class;
                uart_result_expected_class <= expected_class;
                uart_result_confidence <= confidence;
            end

            if (!uart_packet_active) begin
                if (uart_reset_pending) begin
                    uart_packet_active <= 1'b1;
                    uart_packet_is_result <= 1'b0;
                    uart_packet_index <= 3'd0;
                    uart_reset_pending <= 1'b0;
                end else if (uart_result_pending) begin
                    uart_packet_active <= 1'b1;
                    uart_packet_is_result <= 1'b1;
                    uart_packet_index <= 3'd0;
                    uart_result_pending <= 1'b0;
                end
            end else if (!uart_busy) begin
                uart_start_tx <= 1'b1;
                if (uart_packet_index == 3'd6)
                    uart_packet_active <= 1'b0;
                else
                    uart_packet_index <= uart_packet_index + 3'd1;
            end
        end
    end

    always @(*) begin
        case (uart_packet_index)
            3'd0: uart_frame_byte = 8'hAA;
            3'd1: uart_frame_byte = uart_packet_is_result ? 8'h50 : 8'h52;
            3'd2: uart_frame_byte = uart_packet_is_result ? {1'b0, uart_result_sample_index} : 8'd0;
            3'd3: uart_frame_byte = uart_packet_is_result ? {7'd0, uart_result_pred_class} : 8'd0;
            3'd4: uart_frame_byte = uart_packet_is_result ? {7'd0, uart_result_expected_class} : 8'd0;
            3'd5: uart_frame_byte = uart_packet_is_result ? uart_result_confidence : 8'd0;
            default: uart_frame_byte = 8'h55;
        endcase
    end

    uart_tx_module uart_tx_inst (
        .clk(clk),
        .rst_n(rst_n_int),
        .start_tx(uart_start_tx),
        .data_in(uart_frame_byte),
        .uart_tx(uart_tx),
        .busy(uart_busy)
    );

    always @(posedge clk or negedge rst_n_int) begin
        if (!rst_n_int) begin
            scan_div <= 16'd0;
            scan_sel <= 3'd0;
        end else begin
            scan_div <= scan_div + 16'd1;
            if (scan_div == 16'd0)
                scan_sel <= scan_sel + 3'd1;
        end
    end

    always @(*) begin
        an = 8'b1111_1111;
        dp = 1'b1;
        case (scan_sel)
            3'd0: begin an = 8'b1111_1110; digit_val = cycle_counter[3:0]; end  // RIGHTMOST: Counter (0-10)
            3'd1: begin an = 8'b1111_1101; digit_val = 4'hB; end  // Blank space separator
            3'd2: begin an = 8'b1111_1011; digit_val = ((displayed_confidence > 8'd99) ? 8'd99 : displayed_confidence) % 10; end  // Confidence ones
            3'd3: begin an = 8'b1111_0111; digit_val = ((displayed_confidence > 8'd99) ? 8'd99 : displayed_confidence) / 10; end  // Confidence tens
            3'd4: begin an = 8'b1110_1111; digit_val = 4'hB; end  // Blank space separator
            3'd5: begin an = 8'b1101_1111; digit_val = displayed_pred_class ? 4'd1 : 4'd0; end  // Predicted class
            3'd6: begin an = 8'b1011_1111; digit_val = (displayed_pred_class != displayed_expected_class) ? 4'hE : 4'hC; end  // E=Error, C=Correct
            3'd7: begin an = 8'b0111_1111; digit_val = displayed_expected_class ? 4'd1 : 4'd0; end  // Expected class
            default: begin an = 8'b1111_1111; digit_val = 4'hB; end
        endcase
    end

    always @(*) begin
        case (digit_val)
            4'h0: seg = 7'b1000000;
            4'h1: seg = 7'b1111001;
            4'h2: seg = 7'b0100100;
            4'h3: seg = 7'b0110000;
            4'h4: seg = 7'b0011001;
            4'h5: seg = 7'b0010010;
            4'h6: seg = 7'b0000010;
            4'h7: seg = 7'b1111000;
            4'h8: seg = 7'b0000000;
            4'h9: seg = 7'b0010000;
            4'hA: seg = 7'b0001000;
            4'hB: seg = 7'b1111111;
            4'hC: seg = 7'b1000110;
            4'hD: seg = 7'b1000001;
            4'hE: seg = 7'b0000110;
            4'hF: seg = 7'b0001110;
            default: seg = 7'b1111111;
        endcase
    end

endmodule

// ============================================================================
// Inference Core (streaming byte inference engine)
// ============================================================================

module inference_core (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       start,
    input  wire [7:0] sample_byte,
    input  wire       sample_valid,
    input  wire       sample_last,
    output reg [31:0] logits_0,
    output reg [31:0] logits_1,
    output reg        predicted_class,
    output reg [7:0]  confidence,
    output reg        done
);
    localparam integer Q = 12288; // 49152/4

    reg        active;
    reg        finalize_pending;
    reg [31:0] byte_idx;
    reg [31:0] se, so, q0, q1, q2, q3, hi, lo;
    reg [31:0] margin;

    wire tree_pred;
    wire [7:0] tree_conf;

    tree_classifier tree_cls_inst (
        .se(se),
        .so(so),
        .q0(q0),
        .q1(q1),
        .q2(q2),
        .q3(q3),
        .hi(hi),
        .lo(lo),
        .pred(tree_pred),
        .conf(tree_conf)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active <= 1'b0;
            finalize_pending <= 1'b0;
            byte_idx <= 32'd0;
            se <= 32'd0; so <= 32'd0;
            q0 <= 32'd0; q1 <= 32'd0; q2 <= 32'd0; q3 <= 32'd0;
            hi <= 32'd0; lo <= 32'd0;
            logits_0 <= 32'd0;
            logits_1 <= 32'd0;
            predicted_class <= 1'b0;
            confidence <= 8'd0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;

            if (start) begin
                active <= 1'b1;
                finalize_pending <= 1'b0;
                byte_idx <= 32'd0;
                se <= 32'd0; so <= 32'd0;
                q0 <= 32'd0; q1 <= 32'd0; q2 <= 32'd0; q3 <= 32'd0;
                hi <= 32'd0; lo <= 32'd0;
            end

            if (active && sample_valid) begin
                if (byte_idx[0]) so <= so + sample_byte;
                else se <= se + sample_byte;

                if (byte_idx < Q) q0 <= q0 + sample_byte;
                else if (byte_idx < (2*Q)) q1 <= q1 + sample_byte;
                else if (byte_idx < (3*Q)) q2 <= q2 + sample_byte;
                else q3 <= q3 + sample_byte;

                if (sample_byte > 8'd127) hi <= hi + 32'd1;
                if (sample_byte < 8'd64)  lo <= lo + 32'd1;

                byte_idx <= byte_idx + 32'd1;
                if (sample_last) begin
                    active <= 1'b0;
                    finalize_pending <= 1'b1;
                end
            end

            if (finalize_pending) begin
                finalize_pending <= 1'b0;
                predicted_class <= tree_pred;
                confidence <= tree_conf;

                logits_0 <= se;
                logits_1 <= so;
                margin = (se > so) ? (se - so) : (so - se);
                if (margin > 32'd300000)
                    confidence <= 8'd95;
                else if (margin > 32'd150000)
                    confidence <= 8'd80;
                done <= 1'b1;
            end
        end
    end

endmodule

module tree_classifier (
    input  wire [31:0] se,
    input  wire [31:0] so,
    input  wire [31:0] q0,
    input  wire [31:0] q1,
    input  wire [31:0] q2,
    input  wire [31:0] q3,
    input  wire [31:0] hi,
    input  wire [31:0] lo,
    output reg         pred,
    output reg  [7:0]  conf
);
    always @(*) begin
        conf = 8'd65;

        if (lo <= 32'd12374) begin
            if (hi <= 32'd23735) begin
                if (lo <= 32'd9186) pred = 1'b1;
                else begin
                    if (lo <= 32'd9320) pred = 1'b0;
                    else begin
                        if (hi <= 32'd20038) begin
                            if (lo <= 32'd9816) begin
                                if (so <= 32'd2688871) pred = 1'b0;
                                else pred = 1'b1;
                            end else pred = 1'b1;
                        end else begin
                            if (q0 <= 32'd1450671) pred = 1'b0;
                            else pred = 1'b1;
                        end
                    end
                end
            end else begin
                if (so <= 32'd3060880) pred = 1'b0;
                else pred = 1'b1;
            end
        end else begin
            if (lo <= 32'd13331) begin
                if (hi <= 32'd18551) pred = 1'b0;
                else pred = 1'b1;
            end else begin
                if (lo <= 32'd19477) begin
                    if (q1 <= 32'd1239714) pred = 1'b0;
                    else begin
                        if (hi <= 32'd16221) pred = 1'b1;
                        else begin
                            if (hi <= 32'd17000) pred = 1'b0;
                            else begin
                                if (lo <= 32'd18188) pred = 1'b1;
                                else pred = 1'b0;
                            end
                        end
                    end
                end else begin
                    if (q0 <= 32'd1198942) begin
                        if (hi <= 32'd10202) begin
                            if (se <= 32'd2141931) begin
                                if (q1 <= 32'd1053654) pred = 1'b1;
                                else pred = 1'b0;
                            end else pred = 1'b1;
                        end else begin
                            if (q1 <= 32'd1157503) pred = 1'b0;
                            else begin
                                if (se <= 32'd2407553) pred = 1'b1;
                                else pred = 1'b0;
                            end
                        end
                    end else pred = 1'b1;
                end
            end
        end
    end
endmodule
// ============================================================================
// UART TX Module
// ============================================================================

module uart_tx_module (
    input wire clk,
    input wire rst_n,
    input wire start_tx,
    input wire [7:0] data_in,
    output reg uart_tx,
    output reg busy
);

    reg [3:0] bit_count;
    reg [9:0] shift_reg;
    reg [15:0] baud_counter;
    localparam BAUD_RATE = 115200;
    localparam CLOCK_FREQ = 100_000_000;
    localparam BAUD_DIV = CLOCK_FREQ / BAUD_RATE;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            uart_tx <= 1'b1;
            busy <= 1'b0;
            bit_count <= 4'd0;
            shift_reg <= 10'h3FF;
            baud_counter <= 16'd0;
        end else if (start_tx && !busy) begin
            shift_reg <= {1'b1, data_in, 1'b0};
            uart_tx <= 1'b0;
            busy <= 1'b1;
            bit_count <= 4'd0;
            baud_counter <= 16'd0;
        end else if (busy) begin
            if (baud_counter >= (BAUD_DIV - 1)) begin
                baud_counter <= 16'd0;
                shift_reg <= {1'b1, shift_reg[9:1]};
                uart_tx <= shift_reg[1];
                if (bit_count == 4'd8) begin
                    busy <= 1'b0;
                    uart_tx <= 1'b1;
                end else begin
                    bit_count <= bit_count + 4'd1;
                end
            end else begin
                baud_counter <= baud_counter + 16'd1;
            end
        end else begin
            uart_tx <= 1'b1;
        end
    end

endmodule
