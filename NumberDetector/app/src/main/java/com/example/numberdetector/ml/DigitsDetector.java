package com.example.numberdetector.ml;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class DigitsDetector {

    //1. specify name of file in assets folder
    private static final String MODEL_PATH="mnist.tflite";
    private final String TAG =this.getClass().getSimpleName() ;

    //Java provides a class ByteBuffer which is an abstraction of a buffer storing bytes.
    //A ByteBuffer operates on a FileChannel which is a byte channel which has a current position.
    //The FileChannel provides methods for reading from and writing to ByteBuffers.
    //5. Create input buffer
    private ByteBuffer inputBuffer=null;

    //6. Output array [batch_size, 10]
    private float[][] mnistOutput=null;

    //7. Specify the output size
    private static final int NUMBER_LENGTH = 10;

    //8. Specify the input size
    private static final int DIM_BATCH_SIZE = 1;
    private static final int DIM_IMG_SIZE_X = 28;
    private static final int DIM_IMG_SIZE_Y = 28;
    private static final int DIM_PIXEL_SIZE = 1;

    //9. Number of bytes to hold a float (32 bits / float) / (8 bits / byte) = 4 bytes / float
    private static final int BYTE_SIZE_OF_FLOAT = 4;

    //2.The term inference refers to the process of executing a TensorFlow Lite
    //model on-device in order to make prediction based on input data.
    // To perform an inference with a TensorFlow Lite model, you must run it through an interpreter.
    //The interpreter uses a static graph ordering and a custom (less-dynamic)
    //memory allocator to ensure minimal load, initialization, and execution latency.

    //In Java, you'll use the Interpreter class to load a model and drive model inference.
    private Interpreter tflite;

    //4. Initialize the interpreter with the model
    public DigitsDetector(Activity activity)
    {
        try{
            tflite=new Interpreter(loadModelFile(activity));
            //10. allocate input buffer
            inputBuffer=ByteBuffer.allocateDirect(
                    BYTE_SIZE_OF_FLOAT*DIM_BATCH_SIZE*DIM_IMG_SIZE_X*DIM_IMG_SIZE_Y*DIM_PIXEL_SIZE);

            // read or write data to a buffer in the current hardware's "native" ordering
            // (that is, the order in which the your machine's CPU writes words to memory)
            inputBuffer.order(ByteOrder.nativeOrder());
            //11. allocate output
            mnistOutput=new float[DIM_BATCH_SIZE][NUMBER_LENGTH];
        }
        catch (IOException e)
        {
            Log.e(TAG, "IOException loading the tflite file");
        }
    }
    /**
     * 3.Load the model file from the assets folder
     */

    //Java provides a class called MappedByteBuffer (part of JavaNIO),
    //which helps us dealing with large size files.
    //File content is loaded in virtual memory instead of heap and JVM doesn’t need
    //to call OS specific read/write system calls to read write data in JVM memory.
    //We can also map specific part of a file instead of entire file.
    private MappedByteBuffer loadModelFile(Activity activity) throws IOException
    {
        //public abstract AssetManager getAssets ()
        //Returns an AssetManager instance for the application's package.
        //It is method of Context class
        //public AssetFileDescriptor openFd (String fileName)
        //Open an uncompressed asset by mmapping it and returning an AssetFileDescriptor.
        //It is method of AssetManager class
        //It throws IO Exception
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_PATH);

        //Returns the FileDescriptor that can be used to read the data in the file.
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());

        //We can map file with MappedByteBuffer by getting FileChannel.
        //FileChannel is link for reading, writing and manipulating a file.
        //FileChannel can be access by RandomAccessFile, FileInputStream (for read only)
        //and FileOutputStream (for write only).
        FileChannel fileChannel = inputStream.getChannel();

        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();

        //FileChannel provide map() to map the file. It takes 3 arguments
        //1. Map mode (READ_ONLY, READ_WRITE and PRIVATE)
        //2. Position
        //3. Size
        //After getting MappedByteBuffer we can call get() and put() methods to read and write data.
        //Maps a region of this channel's file directly into memory
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    // 12. create a public method called detectDigit that takes in a Bitmap. In the
    // detectDigit method, we’ll pre-process the image to prepare it for the model,
    // run inference, and then interpret the result.

    /**
     * Take in a bitmap and identify the number drawn on it.
     **/
    public int classify(Bitmap bitmap) {
        if (tflite == null) {
            Log.e(TAG, "Image classifier has not been initialized; Skipped.");
        }
        preprocess(bitmap);
        runInference();
        return postprocess();
    }


    //13.
    private void preprocess(Bitmap bitmap) {
        if (bitmap == null || inputBuffer == null) {
            return;
        }

        // Reset the image data. The position is set to zero and the mark is discarded.
        // Invoke this method before a sequence of channel-write or get operations,
        // assuming that the limit has already been set appropriately.
        inputBuffer.rewind();

        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        long startTime = SystemClock.uptimeMillis();

        // The bitmap shape should be 28 x 28
        int[] pixels = new int[width * height];
        //getPixels() returns the complete int[] array of the source bitmap,
        // so has to be initialized with the same length as the source bitmap's height x width.
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        for (int i = 0; i < pixels.length; ++i) {
            // Set 0 for white and 255 for black pixels
            int pixel = pixels[i];
            // The color of the input is black so the blue channel will be 0xFF.
            //It's a so-called mask. The thing is, you get the RGB value all in one integer,
            //with one byte for each component. Something like 0xAARRGGBB (alpha, red, green, blue).
            //By performing a bitwise-and with 0xFF, you keep just the last part, which is blue
            //0xff: 00000000 00000000 00000000 11111111== 0001
            int channel = pixel & 0xff;

            //if pixel is black, hex code is 0000 which And'ed with 0xff gives 0000
            //now, 0001-0000=1=black
            //IF pixel is white, hex code is 0111 which ANd'ed with 0xff gives 0001
            //now, 0001-0001=0 =white
            inputBuffer.putFloat(0xff - channel);
        }
        long endTime = SystemClock.uptimeMillis();
        Log.d(TAG, "Time cost to put values into ByteBuffer: " + Long.toString(endTime - startTime));

    }

    private int postprocess() {
        for (int i = 0; i < mnistOutput[0].length; i++) {
            float value = mnistOutput[0][i];
            Log.d(TAG, "Output for " + Integer.toString(i) + ": " + Float.toString(value));
            if (value == 1f) {
                return i;
            }
        }
        return -1;
    }

    //14 Run the inputBuffer through the TensorFlow Lite model and load the result in mnistOutput
    private void runInference() {
        tflite.run(inputBuffer,mnistOutput);
    }




}
