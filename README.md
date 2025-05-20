# Autism Spectrum Disorder

## 🧠 Project Title  
**Data-Driven Neurodiversity: Predictive Models for Autism Across Lifespans**

## 📝 Project Overview  
This project aims to provide an intelligent prediction system for Autism Spectrum Disorder (ASD) applicable to all age groups including toddlers, children, adolescents, and adults. Developed using machine learning models and a user-friendly web interface, the system offers an engaging experience through age-specific questionnaires and interactive cognitive games. The goal is to support early detection, increase awareness, and make the prediction accessible for educational and healthcare use.

## 🚀 Features  

- 👤 Age-Based Prediction Interface  
Users can select their age group (Toddler, Child, Adult) to receive a tailored questionnaire and prediction output using trained ML models.

- 📊 Machine Learning Algorithms  
Implements multiple ML models such as:
   - Random Forest
   - Decision Tree
   - XGBoost  
Each model is trained on age-specific balanced and cleaned datasets to ensure reliable predictions.

- 🧠 Autism-Specific Questionnaire  
Each age group receives 10 personalized yes/no questions based on medical datasets to evaluate ASD likelihood.

- 🎮 Engaging Cognitive Games  
Includes memory games, drawing tools, maze games and more to support cognitive development and engagement for users.

- 💡 Results & Interpretation  
After submission, the prediction result is displayed clearly with guidance on interpretation and next steps.

- 📚 Clean Datasets & Preprocessing  
Utilizes balanced, cleaned datasets for toddlers, children and adults, with models trained using Jupyter Notebooks and exported as `.pkl` files.

## 📦 Modules and Descriptions  

| Module Name           | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| **Landing Page**          | Welcome message with autism definition, characteristics, and dropdown for age group selection. |
| **Questionnaire Page**    | Displays 10 age-specific autism-related questions with Yes/No options.     |
| **Result Page**          | Displays ML-based prediction after form submission.                        |
| **Game Hub**              | Features games like maze, memory card, drawing tool and more games for engagement. |
| **Backend ML API**        | Python Flask API serving trained ML models based on age group.             |

## 🛠️ Tech Stack  

- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5  
- **Backend**: Python, Flask  
- **Machine Learning**: Scikit-learn, XGBoost, Pandas, NumPy  
- **Tools Used**: Visual Studio Code, Jupyter Notebook, Thunder Client

## 🔮 Future Scope  
- Add user authentication and session storage  
- Collect anonymous usage data for model improvement  
- Deploy models and website to a public domain (e.g., Vercel/Heroku)  
- Add speech-enabled questionnaire interface  

