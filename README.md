![UTA-DataScience-Logo](https://user-images.githubusercontent.com/98781538/226424191-8cadd40f-3610-4ed5-93b1-9e9072098975.png)


# Deep Learning-Based Classification of Herbarium Images of Rubber Rabbitbrush: Implications for Analyzing Climate Change's Impact on Flowering Times

* This repository holds my attempt to apply a pre-trained Transfer Learning model, VGG16 to *Ericameria nauseosa* specimen data retrived from the Intermountain Regional Herbarium Network.

## Overview

  * The purpose of our work is to understand the impact of climate, more specifically temperature changes on the flowering times in *Ericameria nauseosa*. The specimens were obtained from the *Intermountain Regional Herbarium Network* and include our key indicator of **reproductiveCondition** to help us determine which phenological stage our specimen is in. Along with this, we also have meta-data such as altitude, locality, date and other information. 
  <br><br/>
  * Before discussing our approach, we show an overview of a small sample of the exisiting research and other work that has been done in this specific problem domain as it relates to our machine learning task(s). As for our approach, the methods in this repository formulates the problem as a methodology task, comparing different categorical values present in our existing data, understanding the relevant information and removing, re-organizing and properly categorizing the images with a proper label for further objectives later on. 
  <br><br/>
  * The would-be performance of our model would be quantified by many different measures:  The F1 score, Recall, Accuracy and a Confusion Matrix. However, we fell short on our proposed goal and will have to update our summary and repository at a later date.

## Background and Prior Research

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Phenology, the study of periodic life cycle events in organisms, illustrates how environmental factors impact an organism's life over time. This field also sheds light on how changes in these events can affect ecology, underscoring the importance of understanding and quantifying the interplay between phenological shifts and ecological balance, including impacts on population dynamics and nutrient exchanges (Forrest et al. 2010). The connection between environmental changes, particularly climate change, and plant phenology has been extensively documented. <br><br/>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For instance, record spring temperatures in the eastern United States during 2010 and 2012, peaking at 11.0°C and 10.7°C, led to significantly earlier flowering in some plant species (Ellwood et al. 2013). Similarly, a comprehensive analysis of 542 plant species across 21 European countries from 1971 to 2000 revealed a consistent trend of earlier phenological events (Menzel et al. 2006). Moreover, a recent study in the eastern United States, examining over 100 years of data on 36 plant species, found that plants flowered approximately 2.26 days earlier for each 1°C rise in annual average temperature, with no significant difference in response between native and non-native species (Geissler et al. 2023). This growing body of research underscores the profound impact of climate change on plant phenology and its broader ecological implications.
<br><br/>
<br><br/>
<p float="right">
  <img src="https://github.com/JBAguinaga/DATA-4381_Capstone/assets/98781538/3d990440-c843-413e-b177-74ae304a29c2" width="333" height ="300" />
  <img src="https://github.com/JBAguinaga/DATA-4381_Capstone/assets/98781538/60a908f2-fc05-4c86-8077-31390a42f80e" width="333" height ="300" /> 
  <img src="https://github.com/JBAguinaga/DATA-4381_Capstone/assets/98781538/040c5c2d-4d33-49e5-a409-f2d25359cc82" width="333" height ="300" /> 
</p>
<h5> Figure 1, 2, 3: Image of rubber rabbitbrush, native to the Intermountain region. Outline of the Intermountain region from a map point of view. Heatmap that displays temperature increases throughout the globe across 50+ years. </h5>
<br><br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;These type of studies give us insights that heavily rely on herbariums, especially in phenology. Herbariums, usually have records of characteristics of the individual plant such as young flower buds, senescing leaves, and bare branches along with meta-data about the information obtained. They're usually found inside online databases hosted by the Herbarium. However, a pertinent issue is that online herbarium specimens usually need phenological classification in the digitization process, which is resource and time intensive (Ellwood et al. 2019) but are crucial for understanding climate change’s impacts on phenological shifts. <br><br/>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;An effective digitization process involves specimen curation, image capture, image processing, electronic data capture and georeferencing locality descriptions. Since not every institution is taking a standardized approach, problems with the quality of our digitization emerge down the road (Nelson et al. 2012) and although insufficient funds at the local, national, and international scale to match the available workload have led to successful efforts to utilize public participation (Ellwood et al. 2018) there is still room for improvement in the domain of digitized specimen annotation. 
<br><br/>
<br><br/>
<p float="right">
<img src="https://github.com/JBAguinaga/DATA-4381_Capstone/assets/98781538/27e749ff-bff3-4f60-961d-031080dc3a75" width="300" height ="300" />
<img src="https://github.com/JBAguinaga/DATA-4381_Capstone/assets/98781538/7c9ca7c9-0fbe-410a-b836-fb4fac9d9be1" width="300" height ="300" />
<img src="https://github.com/JBAguinaga/DATA-4381_Capstone/assets/98781538/947fb5e2-5ec4-4c19-b5b8-7425bf0355e6" width="400" height ="300" />
<h5> Figure 4, 5, 6: Machine used for photographing phenological images. A sample specimen corresponding to the format of the images we download. An overview of a typical digitized specimen photograph and the different parts of the digital image </h5>
</p>
<br><br/>
&nbsp;&nbsp;&nbsp;&nbsp;Our target, rubber rabbitbrush, is valuable to provide research insights precisely for these reasons. Our objectives for our work are to use online labeled herbarium images, expand upon that data using transfer learning methodologies and the error associated with them, allowing us to perform statistical analysis.  
<br><br/>  
  
## Summary 

### Data

  * Type: 
    * Main: *occurences.csv*:   An excel file that has the available phenological truth values for all the available specimens. These also include other values like the specimen
    * *multimedia.csv*:   Contains all of our image URIs (URLs) 
    * *measurementOrFact.csv*:   Has more specimen entries along with some truth values. However, substantially far less than *occurences.csv*.

  * Size: Total: ~ 10GB
  * Instances: ~ 13,000 specimens (in main excel file)

#### Preprocessing / Clean up
*  

*  

*  

*  

#### Data Visualization
  
    

* Figure 1: 
  
    

* Figure 2: 
   
  
    
### Problem Formulation

  * Input: 
  * Output: 
  
  * Model(s): 
    * *Clustering*: 
    
    * Metric: **  : 
      
        
  
  
  
  
  
### Software Used  

* Packages: 

    * Include the regular "stack": NumPy, Matplotlib, Pandas, Sklearn, Tensorflow, etc.
    
    * Misc: os, random


### Findings/Conclusions

* 
  
*. 

### Future Objectives

* -

* -
* -
## How to reproduce
 
 * Import neccessary packages:
   ```
   import numpy as np
   import pandas as pd
   import sklearn
   ```

 * Download dataset from hyperlink below. The problem is better approached via Google Colaboratory, so all code is tailored for usage on that particular platform. However, the general approach could also be done on another platform such as Kaggle.
 
 * Use general code provided. However, be aware that file paths will be different depending on where dataset is initially loaded onto Google Colaboratory.

### Files in repository

* *image_preprocessing.ipynb*: 
  * Walks through the data collection, preliminary exploratory analysis of the image data, cleaning, organizing and formatting of the relevant image data.

* *Encoding_and_Visualization.ipynb*: 
  * Notebook that provides code for one-hot encoding along with multiple visualizations/plotting of attributes to help build intuition.

### Software Setup

*


### Data

* Download link: 

* *.csv* 


## Citations

&nbsp;&nbsp;&nbsp;&nbsp;Ellwood ER, Temple SA, Primack RB, Bradley NL, Davis CC. 2013. Record-Breaking Early Flowering in the Eastern United States. Hérault B, editor. PLoS ONE. 8(1):e53788. doi:https://doi.org/10.1371/journal.pone.0053788.
<br><br/>
&nbsp;&nbsp;&nbsp;&nbsp;Ellwood ER, Kimberly P, Guralnick R, Flemons P, Love K, Ellis S, Allen JM, Best JH, Carter R, Chagnoux S, et al. 2018. Worldwide Engagement for Digitizing Biocollections (WeDigBio): The Biocollections Community’s Citizen-Science Space on the Calendar. BioScience. 68(2):112–124. doi:https://doi.org/10.1093/biosci/bix143. [accessed 2022 Jun 2]. https://academic.oup.com/bioscience/article/68/2/112/4797259?login=true.
<br><br/>
&nbsp;&nbsp;&nbsp;&nbsp;Ellwood ER, Pearson KD, Nelson G. 2019. Emerging frontiers in phenological research. Applications in Plant Sciences. 7(3). doi:https://doi.org/10.1002/aps3.1234. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6426156/.
<br><br/>
&nbsp;&nbsp;&nbsp;&nbsp;Forrest J, Miller-Rushing AJ. 2010. Toward a synthetic understanding of the role of phenology in ecology and evolution. Philosophical Transactions of the Royal Society B: Biological Sciences. 365(1555):3101–3112. doi:https://doi.org/10.1098/rstb.2010.0145.
<br><br/>
&nbsp;&nbsp;&nbsp;&nbsp;Geissler C, Davidson A, Niesenbaum RA. 2023. The influence of climate warming on flowering phenology in relation to historical annual and seasonal temperatures and plant functional traits. PeerJ. 11:e15188–e15188. doi:https://doi.org/10.7717/peerj.15188. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10124540/.
<br><br/>
&nbsp;&nbsp;&nbsp;&nbsp;Lorieul T, Pearson KD, Ellwood ER, Goëau H, Molino J, Sweeney PW, Yost JM, Sachs J, Mata‐Montero E, Nelson G, et al. 2019. Toward a large‐scale and deep phenological stage annotation of herbarium specimens: Case studies from temperate, tropical, and equatorial floras. Applications in Plant Sciences. 7(3):e01233. doi:https://doi.org/10.1002/aps3.1233. [accessed 2019 Dec 16]. https://bsapubs.onlinelibrary.wiley.com/doi/pdf/10.1002/aps3.1233.
<br><br/>
&nbsp;&nbsp;&nbsp;&nbsp;Menzel A, Sparks TH, Estrella N, Koch E, Aasa A, Ahas R, Alm-kübler K, Bissolli P, Braslavská O, Briede a, et al. 2006. European phenological response to climate change matches the warming pattern. Global Change Biology. 12(10):1969–1976. doi:https://doi.org/10.1111/j.1365-2486.2006.01193.x.
<br><br/>
&nbsp;&nbsp;&nbsp;&nbsp;Nelson G, Paul D, Riccardi G, Mast A. 2012. Five task clusters that enable efficient and effective digitization of biological collections. ZooKeys. 209:19–45. doi:https://doi.org/10.3897/zookeys.209.3135.
<br><br/>
&nbsp;&nbsp;&nbsp;&nbsp;Schuettpelz E, Frandsen PB, Dikow RB, Brown A, Orli SS, Peters MM, Metallo A, Funk VA, Dorr LJ. 2017. Applications of deep convolutional neural networks to digitized natural history collections. Biodiversity Data Journal. 5:e21139–e21139. doi:https://doi.org/10.3897/bdj.5.e21139.
<br><br/>
&nbsp;&nbsp;&nbsp;&nbsp;Stucky BJ, Guralnick R, Deck J, Denny EG, Bolmgren K, Walls R. 2018. The Plant Phenology Ontology: A New Informatics Resource for Large-Scale Integration of Plant Phenology Data. Frontiers in Plant Science. 9. doi:https://doi.org/10.3389/fpls.2018.00517.
<br><br/>
&nbsp;&nbsp;&nbsp;&nbsp;Unger J, Merhof D, Renner S. 2016. Computer vision applied to herbarium specimens of German trees: testing the future utility of the millions of herbarium specimen images for automated identification. BMC Evolutionary Biology. 16(1). doi:https://doi.org/10.1186/s12862-016-0827-5.

