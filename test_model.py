from transformers import pipeline
import scipy

model_id = "dhavalgala/mms-tts-ind-train"
synthesiser = pipeline("text-to-speech", model_id) # add device=0 if you want to use a GPU

speech = synthesiser("jawaban kami atas pertanyaan itu mungkin bergantung pada apakah Anda memikirkan tentang makanan laut atau musik. Itu karena “bass” dan “bass” adalah homonim — dua kata (atau lebih) dengan ejaan atau pengucapan yang sama yang memiliki arti berbeda. Saat Anda menemukan homonim seperti “bass” di alam liar, kemungkinan besar Anda menggunakan petunjuk konteks untuk memahami pertanyaan dan menemukan respons yang tepat. Begitu pula dengan Google Terjemahan. Berkat pembelajaran mesin tingkat lanjut, Terjemahan dapat mengurai konteks dan membedakan berbagai homonim. Namun, untuk mencapai titik ini, diperlukan banyak usaha.")

scipy.io.wavfile.write("finetuned_output.wav", rate=speech["sampling_rate"], data=speech["audio"][0])