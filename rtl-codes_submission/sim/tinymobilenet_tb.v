// ============================================================================
// Test Bench for TinyMobileNet-XS FPGA Inference
// ============================================================================

`timescale 1ns / 1ps

module tinymobilenet_tb ();
    
    // ========================================================================
    // Parameters
    // ========================================================================
    
    localparam integer NUM_SAMPLES = 112;
    
    // ========================================================================
    // Test Signals
    // ========================================================================
    
    reg clk;
    reg reset_button;
    reg start_button;

    wire inference_done;
    wire [6:0] seg;
    wire [7:0] an;
    wire dp;
    
    wire uart_tx;
    reg uart_rx;
    
    integer i;
    integer correct_count_tb;
    integer wrong_count_tb;
    integer sample_done_idx;
    integer wait_cycles;
    integer done_count_tb;

    wire [3:0] state_dbg;
    wire [6:0] sample_idx_dbg;
    wire       pred_dbg;
    wire       exp_dbg;
    wire [7:0] conf_dbg;
    wire [31:0] logit0_dbg;
    wire [31:0] logit1_dbg;

    assign state_dbg    = dut.state;
    assign sample_idx_dbg = dut.sample_index;
    assign pred_dbg     = dut.displayed_pred_class;
    assign exp_dbg      = dut.displayed_expected_class;
    assign conf_dbg     = dut.displayed_confidence;
    assign logit0_dbg   = dut.logits_0;
    assign logit1_dbg   = dut.logits_1;
    
    // ========================================================================
    // Instantiate DUT (Device Under Test)
    // ========================================================================
    
    tinymobilenet_top dut (
        .clk(clk),
        .reset_button(reset_button),
        .start_button(start_button),
        .inference_done(inference_done),
        .uart_rx(uart_rx),
        .uart_tx(uart_tx),
        .seg(seg),
        .an(an),
        .dp(dp)
    );
    
    // ========================================================================
    // Clock Generation (100 MHz)
    // ========================================================================
    
    always #5 clk = ~clk;  // 10ns period = 100 MHz
    
    // ========================================================================
    // Test helper: generate a clean synchronous button pulse
    // ========================================================================

    task press_start_button;
        begin
            @(negedge clk);
            start_button = 1'b1;
            repeat (4) @(posedge clk);
            @(negedge clk);
            start_button = 1'b0;
            repeat (2) @(posedge clk);
        end
    endtask

    // ========================================================================
    // Test Stimulus
    // ========================================================================
    
    initial begin
        // Initialize
        clk = 0;
        reset_button = 1;
        start_button = 0;
        uart_rx = 1;  // UART idle
        correct_count_tb = 0;
        wrong_count_tb = 0;
        i = 0;
        sample_done_idx = 0;
        done_count_tb = 0;
        
        // Dump signals to VCD (waveform) file
        $dumpfile("tinymobilenet_sim.vcd");
        $dumpvars(0, tinymobilenet_tb);
        
        // Reset
        #100 reset_button = 0;
        $display("[%0t] Reset released", $time);
        $display("[INFO] Running %0d samples. Expected sim time is in milliseconds, not microseconds.", NUM_SAMPLES);
        
        // Wait a bit
        #200;
        
        // Run through all samples
        for (i = 0; i < NUM_SAMPLES; i = i + 1) begin
            // Pulse start button
            $display("[%0t] Trigger sample %0d", $time, i);
            press_start_button();

            // Wait for inference to complete (with timeout per sample)
            wait_cycles = 0;
            while ((inference_done !== 1'b1) && (wait_cycles < 2_000_000)) begin
                @(posedge clk);
                wait_cycles = wait_cycles + 1;
            end
            if (wait_cycles >= 2_000_000) begin
                $display("[ERROR %0t] Timeout waiting inference_done at loop i=%0d state=%0d sample_idx=%0d", $time, i, state_dbg, sample_idx_dbg);
                $finish;
            end
            @(posedge clk);
            
            // Get index of the completed sample
            if (dut.sample_index == 0)
                sample_done_idx = NUM_SAMPLES - 1;
            else
                sample_done_idx = dut.sample_index - 1;
            
            // Update correct/wrong counters
            if (pred_dbg == exp_dbg)
                correct_count_tb = correct_count_tb + 1;
            else
                wrong_count_tb = wrong_count_tb + 1;
            
            // Display results
            $display("[%0t] SAMPLE=%0d PRED=%0d ACT=%0d CONF=%0d LOGIT0=%h LOGIT1=%h CORRECT=%0d WRONG=%0d DONE_CNT=%0d STATE=%0d IDX=%0d AN=%b SEG=%b DP=%b",
                     $time,
                     sample_done_idx,
                     pred_dbg,
                     dut.label_rom[sample_done_idx][0],
                     conf_dbg,
                     logit0_dbg,
                     logit1_dbg,
                     correct_count_tb,
                     wrong_count_tb,
                     done_count_tb,
                     state_dbg,
                     sample_idx_dbg,
                     an,
                     seg,
                     dp);
            
            #200;
        end
        
        // Final summary
        $display("---------------------------------------------------------------");
        $display("FINAL RESULT: TOTAL=%0d CORRECT=%0d WRONG=%0d ACC=%.2f%%",
                 NUM_SAMPLES,
                 correct_count_tb,
                 wrong_count_tb,
                 (correct_count_tb * 100.0) / NUM_SAMPLES);
        $display("---------------------------------------------------------------");
        
        // End simulation
        #1000;
        $finish;
    end
    
    // ========================================================================
    // Timeout (prevent infinite simulation)
    // ========================================================================
    
    initial begin
        #250_000_000;  // 250 ms timeout
        $display("[ERROR] Simulation timeout");
        $finish;
    end

    always @(posedge clk) begin
        if (inference_done)
            done_count_tb <= done_count_tb + 1;
    end

endmodule
