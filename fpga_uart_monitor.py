import argparse
import sys
import time
from datetime import datetime

import serial

SOF = 0xAA
EOF = 0x55
FRAME_LEN = 7


def line(char='=', width=84):
    return char * width


def boxed(title, subtitle=None, border='*', width=84):
    inner = width - 4
    print()
    print(line(border, width))
    print(f"{border} {title.center(inner)} {border}")
    if subtitle:
        print(f"{border} {subtitle.center(inner)} {border}")
    print(line(border, width))


def class_text(value):
    return 'PEDESTRIAN WILL MOSTLY CROSS' if value else 'PEDESTRIAN WILL MOSTLY NOT CROSS'


def actual_text(value):
    return 'ACTUAL: PEDESTRIAN WILL CROSS' if value else 'ACTUAL: PEDESTRIAN WILL NOT CROSS'


def match_text(predicted, expected):
    return 'MATCH: CORRECT' if predicted == expected else 'MATCH: WRONG'


def print_reset_banner():
    boxed('RESET DETECTED ON FPGA', datetime.now().strftime('%Y-%m-%d %H:%M:%S'), border='*')
    print(line('_'))
    print(' ' * 18 + 'FPGA RESET COMPLETE - READY FOR NEXT BUTTON PRESS')
    print(line('_'))


def print_prediction(sample_index, predicted, expected, confidence):
    boxed('INFERENCE RESULT', f'Sample #{sample_index}   Confidence: {confidence}%', border='#')
    print(line('='))
    print(class_text(predicted).center(84))
    print(line('-'))
    print(actual_text(expected))
    print(match_text(predicted, expected))
    print(f'Predicted Class: {predicted}')
    print(f'Expected Class : {expected}')
    print(line('='))


def decode_frame(frame):
    if len(frame) != FRAME_LEN or frame[0] != SOF or frame[-1] != EOF:
        return None
    event_type = chr(frame[1])
    return {
        'type': event_type,
        'sample_index': frame[2],
        'predicted': frame[3] & 0x01,
        'expected': frame[4] & 0x01,
        'confidence': frame[5],
    }


def monitor(port, baudrate):
    ser = serial.Serial(port=port, baudrate=baudrate, timeout=0.2)
    print(f'Listening on {port} @ {baudrate} baud... Press Ctrl+C to stop.')

    buffer = bytearray()
    try:
        while True:
            chunk = ser.read(1)
            if not chunk:
                continue

            byte = chunk[0]
            if not buffer:
                if byte == SOF:
                    buffer.append(byte)
                continue

            buffer.append(byte)
            if len(buffer) < FRAME_LEN:
                continue

            frame = bytes(buffer[:FRAME_LEN])
            buffer.clear()
            event = decode_frame(frame)
            if not event:
                continue

            if event['type'] == 'R':
                print_reset_banner()
            elif event['type'] == 'P':
                print_prediction(
                    event['sample_index'],
                    event['predicted'],
                    event['expected'],
                    event['confidence'],
                )
    except KeyboardInterrupt:
        print('\nStopped UART monitor.')
    finally:
        ser.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Monitor FPGA UART events and print readable terminal output.')
    parser.add_argument('--port', default='COM11', help='Serial port, e.g. COM11')
    parser.add_argument('--baudrate', type=int, default=115200, help='UART baud rate')
    args = parser.parse_args()
    monitor(args.port, args.baudrate)
