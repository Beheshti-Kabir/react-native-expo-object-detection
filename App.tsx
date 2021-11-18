import { Camera } from 'expo-camera'
import { cameraWithTensors } from '@tensorflow/tfjs-react-native';
import { Platform } from 'expo-modules-core';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import * as tf from '@tensorflow/tfjs'
import Canvas from 'react-native-canvas';
import { StatusBar } from 'expo-status-bar';
import React, { useEffect, useRef, useState } from 'react';
import { Dimensions, LogBox, StyleSheet, Text, View } from 'react-native';


const TensorCamera = cameraWithTensors(Camera);

const {width, height} = Dimensions.get('window')

LogBox.ignoreAllLogs(true);


export default function App() {
  const[model,setModel] = useState<cocoSsd.ObjectDetection>();
  let getContext = useRef<CanvasRenderingContext2D>();
  let canvas = useRef<Canvas>();

  let textureDims = 
    Platform.OS == 'android' 
      ? {height:1920, width:1080}
      : {height:1200, width:1600};

  function handleCameraStream(images : any){
    const loop = async() => {
      const nextImageTensor = images.next().value;
      if(!model || !nextImageTensor) 
        throw new Error('No model or Image Tensor');
      model
        .detect(nextImageTensor)
        .then((prediction) => {
        // we will draw the rectangle 
        drawRectangle(prediction, nextImageTensor);
      }).catch((error) =>{
        console.log(error);
      });

      requestAnimationFrame(loop);
    };
    loop();
  }

  function drawRectangle(
    predictions: cocoSsd.DetectedObject[], 
    nextImageTensor:any){
      if(!getContext.current || !canvas.current) return;

      const scaleWidth = width / nextImageTensor.shape[1];
      const scaleheight = height / nextImageTensor.shape[0];

      const flipHorizontal = Platform.OS == 'android' ? false : true;
      
      // we will clear previoud prediction
      
      getContext.current.clearRect(0,0,width,height);

      // draw the rectangle for each prediction
      for (const prediction of predictions){
        const[x, y, width, height] = prediction.bbox;
        const boundingBoxX = flipHorizontal
        ? canvas.current.width - x * scaleWidth - width * scaleWidth
        : x* scaleWidth;
        const boundingBoxY = y* scaleheight;

        getContext.current.strokeRect(boundingBoxX,
          boundingBoxY,
          width* scaleWidth,
          height* scaleheight
        );

        getContext.current.strokeText(
          prediction.class,
          boundingBoxX - 5,
          boundingBoxY - 5
        );
      }

  }

  async function  handleCanvas(can : Canvas){
    if(can) {
      can.width = width;
      can.height = height;
      const ctx: CanvasRenderingContext2D = can.getContext('2d')
      ctx.strokeStyle = 'red'
      ctx.fillStyle = 'red'
      ctx.lineWidth = 2 

      getContext.current = ctx;
      canvas.current = can;

    }

  }

  useEffect(() => {
    (async() => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      await tf.ready();
      setModel(await cocoSsd.load());
    })();
  },[])
  
  return (
    <View style={styles.container}>
      <TensorCamera 
        style={styles.camera}
        type={Camera.Constants.Type.back}
        cameraTextureHeight={textureDims.height}
        cameraTextureWidth={textureDims.width}
        resizeHeight={200}
        resizeWidth={152}
        resizeDepth={3}
        onReady={handleCameraStream}
        autorender={true}
        useCustomShadersToResize={false}
      />
      <Canvas style={styles.canvas} ref = {handleCanvas} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  camera : {
    width : '100%',
    height : '100%'
  },
  canvas: {
    position:'absolute',
    zIndex : 10000000,
    width : '100%',
    height : '100%'
  },
});
