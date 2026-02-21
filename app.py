from gettext import translation

from flask import Flask,render_template,request
from transformers import MarianMTModel, MarianTokenizer

app=Flask(__name__)

#to load the model and tokenizer
model_name='Helsinki-NLP/opus-mt-en-hi'
model=MarianMTModel.from_pretrained(model_name)
tokenizer=MarianTokenizer.from_pretrained(model_name)

#create method for translation
def translation(data):
    #convert input text to tensor
    inputs=tokenizer(data,return_tensors='pt',padding=True)

    #generate the translated text
    translated_token=model.generate(**inputs)

    #decode the translated text
    output=tokenizer.decode(translated_token[0],skip_special_tokens=True)
    return output

@app.route('/',methods=['GET','POST'])
def index():
    translated_text="" 
    #to get the input from frontend usin request
    if request.method=='POST':
        data=request.form['data']
        translated_text=translation(data)
    return render_template('index.html',translated_text=translated_text)

if __name__ =='__main__':
    app.run(debug=True)