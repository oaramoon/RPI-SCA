import sys
import pyvisa as visa
import time


class Oscilloscope():
    def __init__(self):
        #### Connecting to the Scope #####
        rm = visa.ResourceManager('@py')
        instruments = rm.list_resources()

        usb = list(filter(lambda x: 'MS5' in x, instruments))
        if len(usb) != 1:
            print('Bad instrument list', instruments)
            sys.exit(-1)

        self.scope = rm.open_resource(usb[0])
        self.scope.timeout = 50000
        self.scope.chunk_size = 1024000
        print("Talking to ", self.scope.query("*IDN?"))

    def release(self):
        self.scope.close()
    
    def query(self,msg):
        return self.scope.query(msg)

    def write(self,msg):
        return self.scope.write(msg)
    
    def save_setup(self):
        self.scope.write(":SAVE:SETup C:\setup0.stp ")

    def load_setup(self):
        self.scope.write(":LOAD:SETup C:\setup0.stp ")

    def save_waveform(self,name,wait_time):
        self.scope.write(":SAVE:WAVeform D:\\"+str(name)+".csv")
        time.sleep(wait_time)
        print("Saved the waveform")

    def single(self):
        self.scope.write(":SINGle")



