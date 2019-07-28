import serial
import time


ser = serial.Serial('COM5',9600)

ser.write('2')

ser.close()

