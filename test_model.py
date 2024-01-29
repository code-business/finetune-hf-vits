from transformers import pipeline
import scipy

# model_id = "dhavalgala/mms-tts-ind-train"
model_id = "dhavalgala/mms-tts-mar-train"
synthesiser = pipeline("text-to-speech", model_id) # add device=0 if you want to use a GPU

# speech = synthesiser("jawaban kami atas pertanyaan itu mungkin bergantung pada apakah Anda memikirkan tentang makanan laut atau musik. Itu karena “bass” dan “bass” adalah homonim — dua kata (atau lebih) dengan ejaan atau pengucapan yang sama yang memiliki arti berbeda. Saat Anda menemukan homonim seperti “bass” di alam liar, kemungkinan besar Anda menggunakan petunjuk konteks untuk memahami pertanyaan dan menemukan respons yang tepat. Begitu pula dengan Google Terjemahan. Berkat pembelajaran mesin tingkat lanjut, Terjemahan dapat mengurai konteks dan membedakan berbagai homonim. Namun, untuk mencapai titik ini, diperlukan banyak usaha.")

speech = synthesiser('या प्रश्नाचे आमचे उत्तर कदाचित तुम्ही सीफूड किंवा संगीताबद्दल विचार करत आहात यावर अवलंबून आहे. कारण “बास” आणि “बास” हे समानार्थी शब्द आहेत — दोन शब्द (किंवा अधिक) ज्यांचे शब्दलेखन किंवा उच्चार भिन्न आहेत. जेव्हा तुम्हाला जंगलात "बास" सारखे समानार्थी शब्द आढळतात, तेव्हा तुम्ही प्रश्न समजून घेण्यासाठी आणि योग्य प्रतिसाद शोधण्यासाठी संदर्भ संकेत वापरत असाल. त्याचप्रमाणे Google Translate सह. प्रगत मशीन लर्निंगबद्दल धन्यवाद, भाषांतर हे संदर्भाचे विश्लेषण करू शकते आणि समानार्थी शब्दांमध्ये फरक करू शकते. मात्र, इथपर्यंत पोहोचण्यासाठी खूप प्रयत्न करावे लागले.')

scipy.io.wavfile.write("./output/finetuned_mar.wav", rate=speech["sampling_rate"], data=speech["audio"][0])