import serial
import time

ser = serial.Serial('COM40',9600)
time.sleep(2)

for i in range(10000):
    ser.write('s')

ser.write('s')
ser.close()

