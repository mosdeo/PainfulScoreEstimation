# PainfulScoreEstimation

## Demo video

<style>
/*---------這如果布景本身沒有在加就好---------*/
body{ 
    padding: 0;
    margin: 0;
}

*{
    -webkit-box-sizing: border-box;
    -moz-box-sizing: border-box;
     box-sizing: border-box;
}
/*---------這如果布景本身沒有在加就好 end---------*/

.bob-container{
  margin: 0 auto;
  padding-left: 15px;
  padding-right: 15px;
}

.bob-row{
  margin-right: -15px;
  margin-left: -15px;
}

.bob-row:before,.bob-row:after{
  display: table;
  content: " ";  
}

.bob-row:after{
  clear: both;
}

/*2個並排 不管在任何尺寸都會2個並排*/
.bob-2item{
  width:50%;
}

/*3個並排 在767px以下會垂直排列*/
.bob-3item{
  width: 100%;
}
/*4個並排 在767px以下會2個排列*/
.bob-4item{
  width:50%;
}

.bob-2item,.bob-3item,.bob-4item{
  position: relative;
  min-height: 1px;
  padding-right: 15px;
  padding-left: 15px;
  float:left;
}

.bob-2item img,.bob-3item img,.bob-4item img{
  width:100%;
  display: block;
}

@media (min-width: 768px) {
  .bob-container {
    width: 750px;
  }
  .bob-3item{
    width: 33.33333333%;
  }
  .bob-4item{
    width:25%;
  }
}
@media (min-width: 992px) {
  .bob-container {
    width: 970px;
  }
}
@media (min-width: 1200px) {
  .bob-container {
    width: 1170px;
  }
}

.img-square {
  object-fit: cover;
  width:230px;
  height:230px;
}
</style>

<div class="bob-container"> 
	<div class="bob-row">
    	<div class="bob-2item">
      		<img class="img-square" src="doc\北韓舉重_Moment.jpg">
      		<div style="height:43px;">Weight Lifting</div>
    	</div>
		<div class="bob-2item">
			<img class="img-square" src="doc\娘家.png">
			<div style="height:43px;">Childbirth</div>
		</div>
	</div>
</div>

- Weight Lifting https://youtu.be/B5e_udgH9D8
- Childbirth https://youtu.be/sonvJNeTp9Q
- Self Test https://youtu.be/hP4KmjBfKqI

## Estimation VS Ground Truth in a sample

![EstimationVSGroundTruth](doc/EstimationVSGroundTruth.jpg)

## Begins at my Master's Thesis

- Thesis PDF in National Central Library [Link](http://handle.ncl.edu.tw/11296/ndltd/22213658258720259065)
- Slide Share [Link](https://www.slideshare.net/LinKaoYuan/ss-65635578)

## Training

- Training Data: [The UNBC-McMaster Shoulder Pain Expression Archive Database](http://www.pitt.edu/~emotion/um-spread.htm)
- Training Devivce/Cost: i5-4210M/8G RAM with 6hours/200epochs

## My build enviroment

- EmguCV（libemgucv-windesktop-3.4.1.2976）
- Microsoft Visual Studio Community 2017 Version 15.6.6
- Microsoft .NET Framework Version 4.7.03056

## Executable/Binaries

- [Release of the repo](https://github.com/mosdeo/PainfulScoreEstimation/releases)
- Runtime dependence
  - A Camera
  - Microsoft .NET Framework Version 4.6.1 above