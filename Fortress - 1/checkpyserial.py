import serial
import time

ser = serial.Serial('COM40',9600)

ser.write('s')

ser.close()

0