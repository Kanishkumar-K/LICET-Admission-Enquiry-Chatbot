from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QListWidgetItem
import random
import nltk
from PyQt5.QtCore import QPropertyAnimation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyQt5.QtGui import QLinearGradient, QColor
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QEasingCurve,QAbstractAnimation
from PyQt5.QtGui import QBrush, QColor, QPalette



class ChatbotGUI(QWidget):
    def __init__(self):
        super().__init__()
        
        # Set up the GUI
        self.setWindowTitle('LICET Admission Enquiry Bot')
        self.setFixedSize(500,800)
        
        # Set up the chat history label
        self.setStyleSheet("border: 1px solid black; padding: 5px; color:#000000; font-weight:italics; background-image: url('image12.png'); background-repeat: no-repeat; background-position: center; background-attachment: fixed;")
        self.chat_history = QLabel(self)
                
        font = QFont("Arial", 10)
        palette = QPalette()
        palette.setColor(QPalette.WindowText, Qt.black)

        style_sheet = "background-color: #f2f2f2; border: 1px solid #ccc;"

        self.chat_history.setFont(font)
        self.chat_history.setPalette(palette)
        self.chat_history.setStyleSheet(style_sheet)


        self.chat_history.setText("\nWelcome to LICET Chatbot! How can i assist you? \n\n")
  
        self.chat_history.setAlignment(Qt.AlignTop)
        self.chat_history.setWordWrap(True)
        self.chat_history.setFont(QFont("Open Sans", 12))
        self.chat_history.setStyleSheet("""
            background-color: #fff;
            border-radius: 10px;
            padding: 10px;
            border: 1px solid #ccc;
        """)

        # Add a scroll area to the chat history label
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.chat_history)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setStyleSheet("""
            QScrollBar:vertical {
                border: none;
                background-color: #f2f2f2;
                width: 10px;
                margin: 0px 0 0px 0;
            }

            QScrollBar::handle:vertical {
                background-color: #888;
                min-height: 50px;
            }

            QScrollBar::add-line:vertical {
                background: none;
            }

            QScrollBar::sub-line:vertical {
                background: none;
            }
            QScrollBar {
                background: #FFFFFF;
            }
        """)
        
        # Set up the user input box and submit button
        self.user_input = QLineEdit(self)
        self.user_input.setStyleSheet('''
        QLineEdit {
        background-image: url();
        background-color: #f2f2f2;
        border: 2px solid #ccc;
        border-radius: 10px;
        padding: 5px;
        font-size: 12px;
        }
        
          QLineEdit:hover {
            background-color: #c2c4c0;
            border: 2px solid #aaa;
        }
        ''')

        self.user_input.setFont(QFont("Open Sans", 11, weight=QFont.Normal))
        self.user_input.setPlaceholderText('Type here:')  # Set the placeholder text for the input box
        self.submit_button = QPushButton('Submit', self)
        self.submit_button.clicked.connect(self.submit_message)
        self.submit_button.setStyleSheet("""
            QPushButton {
                background-color: #006600;
                color: white;
                padding: 10px;
                border: none;
                border-radius: 10px;
                font-size: 14px;
            }
            
            QPushButton:hover {
                background-color: #004d00;
            }
        """)
        
        

        self.close_button = QPushButton('Close', self)
        self.close_button.setFont(QFont("Open Sans", 10, weight=QFont.Normal))
        self.close_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                padding: 10px;
                border: none;
                border-radius: 10px;
                font-size: 14px;
            }
            
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        self.close_button.clicked.connect(self.close)
        # Set up the layout
        vbox = QVBoxLayout()
        vbox.addWidget(scroll)
        hbox = QHBoxLayout()
        hbox.addWidget(self.user_input)
        hbox.addWidget(self.submit_button)
        hbox.addWidget(self.close_button)
        vbox.addLayout(hbox)
        self.setLayout(vbox)
        
        # Initialize the chatbot
        self.responses = {
            "Hi": ["👋👋👋"],
            "What are the admission requirements for your college?": ["To be considered for admission, you will need to submit an application, your official high school Transfer Certificate, and your SSLC or HSC scores. For further details contact our Management [Enter management’s contact details]."],
            "What is the application deadline?": ["The application deadline is [insert deadline date here]. Please make sure to submit your application before this date."],
                    "What is the average class size?": ["The average class size is about 60 students."],
            "Timing": ["The College hours is from 8AM to 4PM"],
            "What about hostel": ["The hostel facilities are very great in LICET. Hoatel accomodation process can be started along with or right after you have completed the Admission process (offline). You can have a view on our hostel and its facilities by looking on to our website. For further more details you can contact the LICET Management or Mr. Suman sir (Hostel Warden contact details)"],
            "What kind of sports teams does the college have?": ["We have a variety of sports teams, including BasketBall, Football, Cricket, Volleyball, Hockey, Tennis, Handball, Table tennis, Badminton, Chess and Athletics. You can find more information about our sports teams on our website."],
            "Can I get academic credit for studying abroad?": ["Yes, you can get academic credit for studying abroad through our study abroad program."],
                  "what is student life like": ["At our college, student life is vibrant and diverse. We have many clubs, organizations, and sports teams for students to get involved in, as well as numerous social events throughout the year. We also have a strong focus on community service and encourage our students to give back to the community.",
                                  "Student life at our college is about more than just academics. We offer a variety of extracurricular activities, clubs, and organizations to help students connect with others who share their interests. We also believe in the importance of community service and offer many opportunities for students to give back."],
            "exit": ["Thank you for your interest in our college. Have a great day!", "Goodbye! We hope to hear from you soon."],
            "bye": ["Thank you for your interest in our college. Have a great day!", "Goodbye! We hope to hear from you soon."]
            }
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(list(self.responses.keys()))
        
    
    def submit_message(self):
        user_input = self.user_input.text()
    
        response = self.process_input(user_input)
        self.add_chat_entry("User: " + user_input, "Bot: " + response)
        self.user_input.clear()
        pulse_animation = QPropertyAnimation(self.submit_button, b"opacity")
        pulse_animation.setDuration(400)
        pulse_animation.setStartValue(0.2)
        pulse_animation.setEndValue(1)
        pulse_animation.setEasingCurve(QEasingCurve.OutQuad)
        pulse_animation.start(QAbstractAnimation.DeleteWhenStopped)
        
    def process_input(self, input_text):
        # Tokenize the input text
        input_tokens = nltk.word_tokenize(input_text.lower())
        
        # Get the TF-IDF vectors for the possible user inputs and the user input
        input_vector = self.vectorizer.transform(input_tokens)
        response_vectors = self.vectorizer.transform(self.responses.keys())
        
        # Calculate the cosine similarities between the input vector and each response vector
        similarities = cosine_similarity(input_vector, response_vectors)
        
        # Get the index of the most similar response
        response_index = similarities.argmax()
        
        # Return the response
        return self.responses[list(self.responses.keys())[response_index]][0]

    def add_chat_entry(self, user_message, bot_message):
        # Append the new chat entry to the chat history label
        self.chat_history.setText(self.chat_history.text() + "\n" + user_message + "\n\n" + bot_message + "\n")



    def add_bot_message(self, text):
        item= QListWidgetItem(text)
        item.setTextAlignment(Qt.AlignRight)
        item.setData(Qt.UserRole, "bot")
        item.setData(Qt.UserRole + 1, True)  # Set is_bot property to True
        self.chat_history.addItem(item)
        
if __name__ == '__main__':
    app = QApplication([])
    gui = ChatbotGUI()
    gui.show()
    app.exec_()