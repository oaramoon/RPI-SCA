import RPi.GPIO as GPIO

class Trigger:
    def __init__(self):
        try:
            # Set the GPIO mode to BCM
            GPIO.setmode(GPIO.BCM)

            # Set GPIO pin 18 as output
            GPIO.setup(1, GPIO.OUT)
            self.trigger_active = True
        except:
            print("Failed to setup GPIO-based trigger - ignoring")
    
    
    def set(self):
        if self.trigger_active == True:
            # Turn on the GPIO pin
            GPIO.output(1, GPIO.HIGH)
    
    def clear(self):
        if self.trigger_active == True:
            # Turn off the GPIO pin
            GPIO.output(1, GPIO.LOW)