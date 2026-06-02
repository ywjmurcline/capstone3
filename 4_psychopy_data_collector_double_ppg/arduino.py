import serial
import time
import csv

# SERIAL_PORT = "/dev/cu.usbmodem1101"

class ArduinoWriter():
    def __init__(self, ppgFileName, core, name, serial_port="/dev/cu.usbmodem1101", port=115200):
        self.ppgFile = open(ppgFileName, 'w', newline='', encoding='utf-8')
        self.ser = serial.Serial(serial_port, port, timeout=0.01)
        time.sleep(2.0)  # many Arduino boards reset when serial opens

        self.writer = csv.writer(self.ppgFile)
        self.writer.writerow(['pc_time_s', 'arduino_time_ms', 'redValue', 'ir_value', 'marker'])

        time.sleep(0.5)
        self.clearCache(core)
        self.ppgFile.flush()

        # open a separate reader
        with open(ppgFileName, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

        # validate
        assert len(rows) > 1, f"{name} No data yet."



    def get_writer(self):
        return self.writer
    
    def get_serial_port(self):
        return self.ser
    
    def serSend(self, content):
        self.ser.write(content)
        self.ser.flush()

    def writerWrite(self, content):
        self.writer.writerow(content)

    def addTag(self, core, serTag, writerTag):
        print("addTag")
        self.serSend(serTag)
        self.writerWrite([core.getTime(), "", "", "", writerTag])

    def clearCache(self, core):
        while self.ser.in_waiting:
            line = self.ser.readline().decode('utf-8', errors='ignore').strip()
            # print(f"line: {line}")
            if not line:
                return
            parts = line.split(',')
            if len(parts) == 4:
                # print(parts)
                arduino_t, red_value, ir_value, marker = parts
                self.writerWrite([core.getTime(), arduino_t, red_value, ir_value, marker]) 

    
    def close(self):
        self.ppgFile.flush()
        self.ppgFile.close()
        self.ser.close()