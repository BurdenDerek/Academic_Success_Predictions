# Academic_Sucess_Predictions
Final project for bootcamp

Presentation:

    Which features carry the most weight in predicting if a student is going to be academicly successful?

    I selected this data because it piqued my interest initially with the title of "Student Alcohol Consumption,"
    but had me locked in when I saw that it also tracked their family life, romantic interests, and how successful their parents are.

    What is the most import for a child to succeed?

    The data in the CSV was already very clean, the only parts that I had to modify were the grade.
    The grades came in as a score out of 20, but I wanted a binary output so I bucketed them into Pass/Fail.

    Extensive Tableau use brought considerable understanding to how the data is distributed,
    and even a peak at which variables might hold the most weight.

Machine Learning Model:

    Preprocessing was very light because of how clean the data is, only bucketed the grades into Pass/Fail.

    With each student having 3 individual grades that we can check we can do something interesting.
    We have the choice to treat the grades from the previous trimesters as additional features,
    and somewhat unsurprising they actually prove to be the greatest predictors of their sucess.

    The data was split into training and testing with just train_test_split().

    So instead of choosing a single model, I was far more interested in testing all of them to see how they compare.
    I used oversampling, undersampling, combination sampling, SMOTEEN, random forest, and easy ensemble.
    If I had to just pick one, I would have to go with using the random forest.
    Random forest's feature importance is so good at being able to give you 
    greater insight into which features mattered the most.

CSVs from Kaggle: https://www.kaggle.com/uciml/student-alcohol-consumption

Context:

    The data were obtained in a survey of students math and portuguese language courses in secondary school. It contains a lot of interesting social, gender and study information about students. You can use it for some EDA or try to predict students final grade.

Content:

    Attributes for both student-mat.csv (Math course) and student-por.csv (Portuguese language course) datasets:

    school      - student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)
    sex         - student's sex (binary: 'F' - female or 'M' - male)
    age         - student's age (numeric: from 15 to 22)
    address     - student's home address type (binary: 'U' - urban or 'R' - rural)
    famsize     - family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)
    Pstatus     - parent's cohabitation status (binary: 'T' - living together or 'A' - apart)
    Medu        - mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
    Fedu        - father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
    Mjob        - mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
    Fjob        - father's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
    reason      - reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other')
    guardian    - student's guardian (nominal: 'mother', 'father' or 'other')
    traveltime  - home to school travel time (numeric: 1 - 1 hour)
    studytime   - weekly study time (numeric: 1 - 10 hours)
    failures    - number of past class failures (numeric: n if 1<=n<3, else 4)
    schoolsup   - extra educational support (binary: yes or no)
    famsup      - family educational support (binary: yes or no)
    paid        - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
    activities  - extra-curricular activities (binary: yes or no)
    nursery     - attended nursery school (binary: yes or no)
    higher      - wants to take higher education (binary: yes or no)
    internet    - Internet access at home (binary: yes or no)
    romantic    - with a romantic relationship (binary: yes or no)
    famrel      - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
    freetime    - free time after school (numeric: from 1 - very low to 5 - very high)
    goout       - going out with friends (numeric: from 1 - very low to 5 - very high)
    Dalc        - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
    Walc        - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
    health      - current health status (numeric: from 1 - very bad to 5 - very good)
    absences    - number of school absences (numeric: from 0 to 93)

These grades are related with the course subject, Math or Portuguese:

    G1 - first period grade (numeric: from 0 to 20)
    G2 - second period grade (numeric: from 0 to 20)
    G3 - final grade (numeric: from 0 to 20, output target)

Additional note: there are several (382) students that belong to both datasets .
These students can be identified by searching for identical attributes
that characterize each student, as shown in the annexed R file.

Source Information
P. Cortez and A. Silva. Using Data Mining to Predict Secondary School Student Performance. In A. Brito and J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008) pp. 5-12, Porto, Portugal, April, 2008, EUROSIS, ISBN 978-9077381-39-7.

Fabio Pagnotta, Hossain Mohammad Amran.
Email:fabio.pagnotta@studenti.unicam.it, mohammadamra.hossain '@' studenti.unicam.it
University Of Camerino

https://archive.ics.uci.edu/ml/datasets/STUDENT+ALCOHOL+CONSUMPTION