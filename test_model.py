from transformers import pipeline
import scipy

# model_id = "dhavalgala/mms-tts-ind-train"
model_id = "dhavalgala/mms-tts-mar-train"
synthesiser = pipeline("text-to-speech", model_id) # add device=0 if you want to use a GPU

# speech = synthesiser("jawaban kami atas pertanyaan itu mungkin bergantung pada apakah Anda memikirkan tentang makanan laut atau musik. Itu karena “bass” dan “bass” adalah homonim — dua kata (atau lebih) dengan ejaan atau pengucapan yang sama yang memiliki arti berbeda. Saat Anda menemukan homonim seperti “bass” di alam liar, kemungkinan besar Anda menggunakan petunjuk konteks untuk memahami pertanyaan dan menemukan respons yang tepat. Begitu pula dengan Google Terjemahan. Berkat pembelajaran mesin tingkat lanjut, Terjemahan dapat mengurai konteks dan membedakan berbagai homonim. Namun, untuk mencapai titik ini, diperlukan banyak usaha.")

speech = synthesiser('भारत, एक जनसंख्या-भरित दक्षिण आशियाचा देश, १९४७ मध्ये स्वतंत्रता मिळवून ऐतिहासिक गोंधळे, विविध प्राकृतिक सौंदर्य, आणि मजबूत आर्थिक प्रणाली दाखवतो. हिंदूधर्म, बौद्धधर्म, आणि विविध भाषांच्या समृद्ध संस्कृतीमध्ये भारत साकारात्मक विविधतेचे उदाहरण आहे. गरीबी, पर्यावरण समस्यांपैकी संघर्ष करता येतंय, परंतु भारत एक वैश्विक आयटी केंद्र आणि सुचलित चित्रपट उद्योगांचं देश आहे.')

scipy.io.wavfile.write("./output/finetuned_mar.wav", rate=speech["sampling_rate"], data=speech["audio"][0])