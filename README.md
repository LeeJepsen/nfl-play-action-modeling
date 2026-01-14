# NFL Play Action Modeling Exploration (Supervised Learning)
This repo contains the full writeup, R scripts, and data file from NFLs 2025 Big Data Bowl plays file.

## Why this project
In the modern NFL, play action remains one of the most efficient pass concepts, and offenses continue to increase their use of pre-snap motion to disguise intent and stress defenses. This project started with a practical football question:

**Can I use pre-snap information to predict when a play action pass will be called, and when it will be successful?** 

Rather than treating this as a “perfect prediction” problem, I approached it as a supervised learning exploration to understand:
- what pre-snap variables actually provide signal
- what factors are likely missing from the dataset
- how different modeling tools perform in a football context

## Project objectives
This project consists of two supervised learning objectives:

### 1) Predict Play Action Calls (Classification)
Predict whether a play will be a **play action pass** using pre-snap variables.

### 2) Predict Play Action Success (Classification)
Predict whether a **play action pass is successful** using game context and defensive coverage.

## Data
- **NFL Big Data Bowl 2025 dataset** (Weeks 1–9 of the 2022 NFL season) 
- Play action plays made up **~17%** of the dataset → large class imbalance 

Key feature groups included:
- Defensive coverage and man/zone structure
- Down/distance and field position variables (engineered)
- Pre-snap win probability / game state context 

## Methods

### Phase 1 — Predicting Play Action Calls
Models tested:
- Decision Tree (Gini index)
- **Random Forest (final model choice)** 

Key challenges:
- Strong class imbalance (play action minority class)
- Initial tree overfit early splits
- Random Forest improved stability but still struggled with specificity

Evaluation:
- Confusion matrix (sensitivity/specificity)
- ROC curve + AUC
- Cumulative gains / lift chart

### Phase 2 — Predicting Play Action Success
**Success metric**
Initially explored EPA, then used a situational success definition:
- 1st down: gain ≥ 40% of yards-to-go  
- 2nd down: gain ≥ 60%  
- 3rd/4th down

Dimensionality reduction:
- **Principal Component Analysis (PCA)** to reduce situational variables into 4 components (98% variance retained)

Model:
- **K-Nearest Neighbors (KNN)**
- 5-fold CV + k tuning
- feature standardization for distance-based learning 

Evaluation:
- ROC curve + AUC
- Decile-wise lift chart 

## Results (What I found)

### Predicting Play Action Calls
- Random Forest Accuracy: **83%**
- AUC: **0.593** (only slightly better than random) 
- The model showed high sensitivity but low specificity and frequently over-predicted play action.

Feature importance suggested the strongest drivers were:
- Defensive coverage
- Run-pass option indicator
- Pre-snap win probability

**Takeaway:** Pre-snap variables alone provide limited signal for play action prediction. The results support the idea that play action is largely a strategic coaching decision influenced by context.


### Predicting Play Action Success
- Best KNN Validation Accuracy: **55.5%**
- AUC: **0.52** (near random) 
- Lift chart results showed the model struggled to rank successful plays above baseline.

**Takeaway:** Pre-snap variables and defensive coverage are not sufficient to predict play action success with meaningful reliability. Success likely depends more on play design, protection, execution, and personnel matchups.  Some play-calling decisions require more context than typical pre-snap datasets provide.

Even though predictive performance was limited, the modeling workflow highlighted what data likely matters most for future model improvement:
- offensive personnel groupings
- formation/alignment
- motion + shifts
- play sequencing and tendency features
- matchup-specific context

## For entire thought process and reasoning check writeup folder which houses full PDF of project.

## Key Visuals

### Random Forest Performance (Play Action Call Prediction)
AUC = 0.593 (slightly better than random), showing limited predictive separation from pre-snap variables alone.
<img width="503" height="389" alt="image" src="https://github.com/user-attachments/assets/1e32a7c1-9504-49a5-8927-879ae9b02d99" />


### Feature Importance (Random Forest)
Coverage and game context provided the strongest directional signal.
<img width="602" height="454" alt="image" src="https://github.com/user-attachments/assets/a940049a-9fbd-4eab-8673-ea7174a57f07" />



### Cumulative Gains / Lift
Model prioritization was only slightly better than a random baseline.
<img width="558" height="339" alt="image" src="https://github.com/user-attachments/assets/47fc00ae-7ac3-4520-bf95-789d7582b2d1" />


