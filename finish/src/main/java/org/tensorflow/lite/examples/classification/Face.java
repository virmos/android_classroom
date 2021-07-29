package org.tensorflow.lite.examples.classification;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class Face {
    private Interpreter interpreter;
    private int INPUT_SIZE;
    private int height = 0;
    private int width = 0;

    private GpuDelegate gpuDelegate = null;

    private CascadeClassifier cascadeClassifier;
    Face(AssetManager assetManager, Context context, String modelPath, int inputSize) throws IOException {
        INPUT_SIZE = inputSize;
        Interpreter.Options options = new Interpreter.Options();
        gpuDelegate = new GpuDelegate();
        options.setNumThreads(2);
        interpreter = new Interpreter(loadModel(assetManager, modelPath), options);
        // when model is successfully loaded

        try {
            InputStream inputStream = context.getResources().openRawResource(R.raw.haarcascade_frontalface_alt);
            // create new folder to save classifier
            File cascadeDir = context.getDir("cascade", Context.MODE_PRIVATE);
            // create cascade file in that folder
            File cascadeFile = new File(cascadeDir, "haarcascade_frontalface_alt");
            FileOutputStream outputStream = new FileOutputStream(cascadeFile);
            byte []buffer = new byte[4096];
            int byteRead;
            // -1 means no data to read
            while ((byteRead=inputStream.read(buffer))!=-1) {
                outputStream.write(buffer, 0, byteRead);
            }
            inputStream.close();
            outputStream.close();

            // load cascade classifier
            cascadeClassifier = new CascadeClassifier(cascadeFile.getAbsolutePath());
        }
        catch(IOException e) {
            e.printStackTrace();
        }
    }

    public Mat recognizeImage(Mat matImage) {
        Core.flip(matImage.t(), matImage, 1);

        Mat grayScaleImage = new Mat();
        Imgproc.cvtColor(matImage, grayScaleImage, Imgproc.COLOR_RGBA2GRAY);
        height = grayScaleImage.height();
        width = grayScaleImage.width();
        // minimum height and width
        int absoluteFaceSize = (int) (height*0.1);
        // store all faces
        MatOfRect faces = new MatOfRect();
        if (cascadeClassifier != null) {
            // detect face in frame
            // faces == output
            cascadeClassifier.detectMultiScale(grayScaleImage, faces, 1.1, 2, 2,
                    new Size(absoluteFaceSize, absoluteFaceSize), new Size());
        }
        Rect[] faceArray = faces.toArray();
        for (int i = 0; i < faceArray.length; i++) {
            // drawRectangle around faces
            //                  input       output             endpoint         RGBA               thickness
            Imgproc.rectangle(matImage, faceArray[i].tl(), faceArray[i].br(), new Scalar(0,255,0,255), 2);
            Rect roi = new Rect((int)faceArray[i].tl().x, (int)faceArray[i].tl().y,
                    (int)faceArray[i].br().x - (int)faceArray[i].tl().x,
                    (int)faceArray[i].br().y - (int)faceArray[i].tl().y
                    );
            // roi is used to crop faces from image
            Mat croppedRGB = new Mat(matImage, roi);
            // convert croppedRGB to bitmap
            Bitmap bitmap = null;
            bitmap = Bitmap.createBitmap(croppedRGB.cols(), croppedRGB.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(croppedRGB, bitmap);
            // scale bitmap to model input size 96
            Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);
            ByteBuffer byteBuffer = convertBitmapToByteBuffer(scaledBitmap);
            float[][] faceValue = new float[1][1];
            interpreter.run(byteBuffer, faceValue);
            // read face value
            float readFace = (float) Array.get(Array.get(faceValue, 0), 0);
            String faceName = getFaceName(readFace);
            // puttext on frame
            //              input           output
            Imgproc.putText(matImage,"" + faceName, new Point((int)faceArray[i].tl().x + 10,
                    (int)faceArray[i].tl().y + 20),
                    1, 1.5, new Scalar(255, 255, 255, 150), 2);
                    // size                             RGBA
        }
        Core.flip(matImage.t(), matImage, 0);
        return matImage;
    }

    private String getFaceName(float readFace) {
        String val = "";
        if (readFace >= 0 & readFace < 0.5) {
            val = "First person";
        }
        else if (readFace >= 0.5 & readFace < 1) {
            val = "Second person";
        }
        return val;
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap scaledBitmap) {
        ByteBuffer byteBuffer;
        // 4 if input model is float
        // 3 if input is RGB
        // if input GRAY 3 -> 1
        byteBuffer = ByteBuffer.allocateDirect(4 * 1 * INPUT_SIZE * INPUT_SIZE * 3);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];
        scaledBitmap.getPixels(intValues, 0, scaledBitmap.getWidth(), 0, 0
                ,scaledBitmap.getWidth()
                ,scaledBitmap.getHeight()
        );
        int pixels = 0;
        for (int i = 0; i < INPUT_SIZE; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                // each pixel value
                final int val = intValues[pixels++];
                // put this pixel value to bytebuffer
                byteBuffer.putFloat(((val >> 16)&0xFF)/255.0f);
                byteBuffer.putFloat(((val >> 8)&0xFF)/255.0f);
                byteBuffer.putFloat(((val >> 0)&0xFF)/255.0f);
                // placing RGB to MSB to LSB

            }
        }
        return byteBuffer;
    }

    private MappedByteBuffer loadModel(AssetManager assetManager, String modelPath) throws IOException {
        // description of model path
        AssetFileDescriptor assetFileDescriptor = assetManager.openFd(modelPath);
        // inputstream to read model path
        FileInputStream inputStream = new FileInputStream(assetFileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = assetFileDescriptor.getStartOffset();
        long declaredLength = assetFileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
}
