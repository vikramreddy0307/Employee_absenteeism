# Employee_absenteeism
					
 # Problem statement
 XYZ is a courier company. 
The company is passing through genuineissue of Absenteeism. 

As it appreciates that human capital plays an important role
in collection, transportation and delivery. 

The company has shared it dataset and requested to 
answer on the following areas:

* What changes company should bring to reduce the number of absenteeism?
* How much losses every month can we project in 2011 if same trend of
absenteeism continues?

 # Data
Dataset Details:
 Number of Attributes: 21 
Missing Values : Yes 
Attribute Information:
*  Individual identification (ID) 
*  Reason for absence (ICD).
 Absences attested by the International Code of Diseases (ICD) stratified into 21 categories (I to XXI) as follows
: I Certain infectious and parasitic diseases 
II Neoplasms
 III Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism 
IV Endocrine, nutritional and metabolic diseases 
V Mental and behavioural disorders
 VI Diseases of the nervous system 
VII Diseases of the eye and adnexa
VIII Diseases of the ear and mastoid process
 IX Diseases of the circulatory system
 X Diseases of the respiratory system 
XI Diseases of the digestive system 
XII Diseases of the skin and subcutaneous tissue 
XIII Diseases of the musculoskeletal system and connective tissue 
XIV Diseases of the genitourinary system 
XV Pregnancy, childbirth and the puerperium 
XVI Certain conditions originating in the perinatal period 
XVII Congenital malformations, deformations and chromosomal abnormalities 
XVIII Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified 
XIX Injury, poisoning and certain other consequences of external causes 
XX External causes of morbidity and mortality 
XXI Factors influencing health status and contact with health services. And 7 categories without (CID) patient follow-up (22), medical consultation (23), blood donation (24), laboratory examination (25), unjustified absence (26), physiotherapy (27), dental consultation (28). 3. Month of absence 4. Day of the week (Monday (2), Tuesday (3), Wednesday (4), Thursday (5), Friday (6)) 5. Seasons (summer (1), autumn (2), winter (3), spring (4))
*  Transportation expense
*  Distance from Residence to Work (kilometers) 
*  Service time 
*  Age 10. Work load Average/day 
*  Hit target 
*  Disciplinary failure (yes=1; no=0) 
*  Education (high school (1), graduate (2), postgraduate (3), master and doctor (4)) 14. Son (number of children) 
*  Social drinker (yes=1; no=0) 
*  Social smoker (yes=1; no=0) 
*  Pet (number of pet) 
*  Weight 
*  Height 
*  Body mass index 
*  Absenteeism time in hours (target)

# Exploratory data analysis
* Recoding factor variables
* Data exploration
* Percentage distribution of factor variables
* Histogram distribution of continuous variables
* Boxplot graphs of continuous variables
* Type -2 Anova test 
# Data preprocessing
* Missing value imputation
* Outlier removal
* Data scaling
* One hot encoding of factor variables
* Removing correlated features from the data

# Feature Engineering
* Feature selection using Boruta in R
* Feature selection using SelectKBest in python
* Principal component analysis
# Sampling methods
* K-fold repeated CV
