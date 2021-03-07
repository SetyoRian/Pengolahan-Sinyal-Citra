import pyaudio
import wave
import numpy as np
import simpleaudio as sa
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time as tm


def record():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "Audio.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Recording....")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("....Done")
    tm.sleep(1)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def play():
    filename = 'Audio.wav'
    wave_obj = sa.WaveObject.from_wave_file(filename)
    play_obj = wave_obj.play()
    play_obj.wait_done()  # Wait until sound has finished playing


def plot():
    raw = wave.open("Audio.wav")
    signal = raw.readframes(-1)
    signal = np.frombuffer(signal, dtype="int16")
    # gets the frame rate
    f_rate = raw.getframerate()
    # to Plot the x-axis in seconds
    # you need get the frame rate
    # and divide by size of your signal
    # to create a Time Vector
    # spaced linearly with the size
    # of the audio file
    time = np.linspace(
        0,  # start
        len(signal) / f_rate,
        num=len(signal)
    )
    plt.figure(1)
    plt.title("Sound Wave")
    plt.xlabel("Time")
    plt.plot(time, signal)
    plt.show()
    plt.savefig('filename')


def openCam():
    vid = cv2.VideoCapture(0)
    while True:
        ret, frame = vid.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # After the loop release the cap object
    showPic = cv2.imwrite("Picture.jpg", frame)
    print(showPic)
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


def capture():
    camera = cv2.VideoCapture(0)
    return_value, image = camera.read()
    cv2.imwrite('Picture.jpg', image)
    del camera


def showIm():
    img = mpimg.imread('Picture.jpg')
    plt.imshow(img)
    plt.show()


print("Welcome !!!")
while True:
    print("1. Record Audio\n2. Play Audio\n3. Plot Signal Audio")
    print("4. Open Camera\n5. Capture\n6. Show Image\n7. Exit")
    print("Input : ")
    pil = int(input())
    if pil == 1:
        record()
    elif pil == 2:
        play()
    elif pil == 3:
        plot()
    elif pil == 4:
        openCam()
    elif pil == 5:
        capture()
    elif pil == 6:
        showIm()
    elif pil == 7:
        print("Thank You!!")
        exit()
    else:
        print("Invalid Option")
