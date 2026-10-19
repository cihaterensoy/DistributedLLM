"""
bu modül, dil modelinin mimarisini katman katman manuel olarak yürütmekten ve modelin her bir katmanıın bellek/parametre raporunu cıkarmaktan sorumludur
"""

import torch

def manual_gpt2_logits(model,input_ids:torch.Tensor) -> torch.Tensor:
    """
    GPT-2 MODELİNİ KATMAN KATMAN MANUEL OLARAK İLERİYE DOĞRU ÇALIŞTIRACAĞIZ YANİ FORDWARD PASS

    normalde model(input_ids) cağrısını tek seferde çalıştırır
    şimdi biz yöneteceğiz girdi->embedding->block 0 -> block 1-> ... -> block n -> final norm -> lm head tarzında
    """
    #1)girdi boyt kontrolü

    if input_ids.ndim !=2:
        raise ValueError("input_ids must have shape [batch,squence]/girdi boyutu [batch,squence] tarzında olmalıdır")
        #transformer modelelri batch halinde çaşıştığı için ([batch_size,seuence_length]) tek bir cümle gönderilse bile boyut [1,sequence_lenght] olmak zorunda
        #eğer kullanıcı sadece [sequence_length] gönderirse yani tek boyutlu tensör hata döndüreceğiz

    transformer = model.transformer #modelin gövdesi
    batch,sequence = input_ids.shape

    #2)kapasite kontrolü

    if sequence>model.config.n_positions:
        raise ValueError("Sequence exceeds the model's position capacity")
    #gpt2'nin sabit context windowu vardır yani modelin pozisyon tablosunda en fazla n_positions kadar yer vardır
    #verilen cümle bundan uzunsa model hata verir

    #3)pozisyon indekslerini oluşturma
    positions = torch.arange(sequence, device=input_ids.device).unsqueeze(0)
    #transofrmer modelleri kelimelerin sırasını bilmiyor attention mekanizmasında sıra bağımsızdır
    #bu yüzden her kelimenin kaçıncı sırada olduğunu belirten pozisyon tensörü üretilir 4 kelimelik bir girdi için [0, 1, 2, 3]
    # device=input_ids.device ile CPU veya GPU nerede çalışıyorsa oraya yerleştiriyoruz
    # .unsqueeze(0) ile boyutu [4]'ten [1, 4] yapıyoruz (batch boyutuna uyum sağlamak için)

    #4)embedding
    hidden=transformer.wte(input_ids) + transformer.wpe(positions)
    #te (Word Token Embeddings): Kelime ID'lerini (sayıları) gizli vektörlere dönüştürür.
    # Örneğin token ID 5 -> 768 boyutlu bir vektör.
    # wpe (Word Position Embeddings): Pozisyon indekslerini (0, 1, 2...) 768 boyutlu vektörlere dönüştürür.
    # İki vektör toplanır: Böylece kelimenin hem anlamı hem de cümledeki yeri tek bir vektörde birleşir.

    hidden=transformer.drop(hidden)#dropout eğitimde rastgele nöron kapatıyor ama cıkarım kısmında sedece veriyi geçirir

    #5)transformer katmanları
    for block in transformer.h:#gpt2 için 12 katman içerir
        #her blok içinde multi-head attention,layerNorm ve MLP bulunur
        result=block(hidden,use_cache=False)
        hidden = result[0] if isinstance(result, (tuple, list)) else result
        # use_cache=False: Şimdilik KV cache belleğini devre dışı bırakıyoruz, tam hesaplama yapıyoruz.
        # block() bir tuple döner. İlk eleman [0] işlenmiş yeni gizli durumdur (hidden_states)
        # Çıkan sonucu doğrudan bir sonraki bloğun girdisi yapıyoruz

    #6)final norm ve lm head
    hidden=transformer.ln_f(hidden)

    #lm head language model head - linear bir katmandır
    # 768 boyutlu vektörü, kelime haznesi (vocab_size, örn. 50257) boyutuna genişletir.
    # Çıkan değerler "logits"tir (her bir kelimenin olma ham olasılık puanı).
    return model.lm_head(hidden)


def layer_report(model):
    #modelin katmanlarlını inceleyip her bir katmanın parametre sayısını ve ram/vram üzerinde kapladığı yaklasık ağırlık boyutunu mib cinsinsen hesaplar

    report=[]
    #transformer.h içindeki her katmanı indeksleyerek geziyoruz
    for i,block in enumerate(model.transformer.h):
        parametes=list(block.parameters())

        count = sum(p.numel() for p in parametes)# p.numel(): Bir tensörün içindeki toplam eleman (sayı) adedidir.

        nbytes=sum(p.numel()*p.element_size() for p in parametes)
        # Toplam bayt = eleman sayısı * bayt büyüklüğü

        report.append({
            "layer": i,
            "parameters": count,
            "weight_mib": round(nbytes / 2 ** 20, 2)
        })
    return report
