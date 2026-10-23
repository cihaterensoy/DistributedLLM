"""
Model iskeletini meta cihazında oluşturmak, sürekli gereken ağırlıkları ram'e yüklemek ve transformer bloklarınını ağırlıklarını yalnızca blok çalışırken yükleyip ardından meta cihazında geri döndürmek

meta gerçek bir hesaplama için kullanılmaz ağırluık verisi taşımaz
ara aktivasyonlar giris/cıkış katmanlarıve logits yine ram kullanacak
"""

from __future__ import annotations

import threading #aynı nesne üzerinde eşzamanlı forward cağrılarını engellemek icin kullanacağüız
from pathlib import Path #model klasör yolunu ifade etmek için kullanacağız
import torch
from torch import nn

from .adapters import get_adapter #model ailesine uygun katman yolunu bulacak fonksiyonumuz
from .weight_index import WeightIndex #diskteki ağırlık haritasını yönetir

class StreamingCausalLM:
    def __init__(self,model_dir: str | Path, max_layer_bytes: int | None = None):
        """
        model_dir -> config.json ve safetensors dosyalarını bulunduğu klasör
        max_layer_bytes: tek bir transformer bloğunun ağırlık boyutuna isteğe bağlı üst sınır none ise bu sınır uygulanmaz
        yalnızca yüklenen katmanın tensor boyutlarının toplamını denetler toplam ram  kullanımı olarka kullanılmıyor
        :param model_dir:
        :param max_layer_bytes:
        """
        from transformers import AutoConfig, AutoModelForCausalLM

        self.store=WeightIndex(model_dir)#diskteki tensör adları ve dosya eşleşmesi hazırlanır

        config=AutoConfig.from_pretrained(
            str(self.store.folder),
            local_files_only=True,#ağdan dosya indirmeyi engeller dosyalar yerelde olmalı
            trust_remote_code=False,#model deposundaki özel python kodları çalıştırılmaz ancak bazı özel modeller desteklenmeyebilir
        )

        #config içersindeki model_type adnda uygun adaptörü alıyoruz gpt2 iste GPT-2 yolu seçilecek
        self.adapter=get_adapter(config.model_type)


        if max_layer_bytes is not None and max_layer_bytes <= 0:
            raise ValueError("max_layer_bytes pozitif olmalıdır")
        self.max_layer_bytes=max_layer_bytes

        self._lock=threading.Lock() #aynı nesne üzerinde aynı anda iki forward calışırsa katman yükleme durumlarının karışmasını engelleyen kilit

        self._loaded=None #o anda gerçek ağırlıkları yüklenmiş transfrmer katmanıın indeksi, none old için hiç bir ağırlık yüklenmedi

        self.layer_reads=0 # kaç kez bir katman için diskten ağırlık okuma girişiminde bulunuldu

        #1) meta model iskeleti

        with torch.device("meta"):
            self.model=AutoModelForCausalLM.from_config(config,trust_remote_code=False)
            #meta cihazı tensörlerin şeklini tutmaya yarayan ama gerçek sayısal ağırlık verisi taşımayan iskelet kurar
            #böylece rastgele başlangıç ağırlıkları için normal  ram ayrılmasından kacınıllır

        self.model.eval()#dropoutu vs kapattık yani eğitim modunu kapattık, gradient hesaplamasını kapatmaz

        self.layers=self.adapter.layers(self.model)# Adapter üzerinden blokların koleksiyonunu al
        self.prefixes=[self.adapter.prefix(i) for i in range(len(self.layers))]
        # Her blok için state_dict adlarının başlangıç önekini hazırla.
        # Örn. ["transformer.h.0", "transformer.h.1", ...]

        #2)hangi ağırlıklar blok hangileri kalıcı

        all_keys=list(self.model.state_dict().keys())# state_dict anahtarları modeldeki parametre ve buffer adlarını verir

        self.layer_names:list[list[str]]=[]# layer_names[i] -> i numaralı bloğa ait state_dict anahtarları
        block_keys=set()# Bütün Transformer bloklarının ağırlık adlarını tek kümede biriktir
        for prefix in self.prefixes:


            names=[
                name for name in all_keys
                if name.startswith(prefix+".")# Yalnızca "prefix." ile başlayan adları seç
            # Nokta kullanılması, örneğin layer.1 ile layer.10 adlarının yanlışlıkla karşımasını önler
            ]
            if not names:
                raise ValueError(f"{prefix} için state_dict anahtarı bulunamadı")
            # Hiçbir anahtar bulunmaması adaptör yolu ile model yapısının birbiriyle uyuşmadığını söyler

            self.layer_names.append(names)
            block_keys.update(names)

        resident_names=[
            name for name in all_keys if name not in block_keys
        ]
        #transormer blocklarına ait olmayan tüm state_dict adları
        #bunlar genellikle token embedding positional embedding, son norm ve dil modellerinin cıkış başlığı gibi sürekli gerekli olan şeylerdir

        persistent=self.store.load_tensors(
            [name for name in resident_names if name in self.store.keys()],
        )
        #yalnızca hem kalıcı listede bulunan hem de dosya haritasında mevuct olan ağırlıkları diskten oku
        #dosya eksik olan kalıcı anahtarlar burada atlanır, meta kontrol ile eksik kalanlar yakalanabilir

        for key, tensor in persistent.items():
            self._assign_parameter_or_buffer(key, tensor)# Okunan tensorleri meta modelin doğru parametre/buffer alanlarına yerleştir.

        #bazı modeller kelime embedding matrisi ile cıkış ağırlıklarını paylaşır. jmodel configib unu belirtiyorsa bağlantıyı yeniden kur
        if getattr(config,"tie_word_embeddings",False):
            self.model.tie_weights()

        for name in resident_names:
            parameter=self.model.state_dict()[name]
            if parameter.is_meta:
                raise RuntimeError(f"Kalıcı tensor hâlâ meta: {name}")
            #kalıcı olması gerekn her tensorün meta durumunda kalmadığını doğruluyoruz

        #3)forward hooklarını kaydetmek
        # Pre-hook, modülün forward işlemi başlamadan önce çağrılır.
        # Post-hook, modülün forward işlemi bittikten sonra çağrılır.
        self._handles=[]

        for idx,block in enumerate(self.layers):
            self._handles.append(block.register_forward_pre_hook(self._make_pre_hook(idx)))
            #bu bloğun ağırlıklarını çalışmadan hemen çnce yükleyecek callback


            self._handles.append(
                block.register_forward_hook(
                    self._make_post_hook(idx), always_call=True
                )
            )#bloğun ağırlıklarıın forward sonrasında metaya döndürecek callback
            ## always_call=True, desteklenen PyTorch sürümlerinde forward hata versede hookun çağrılmasını sağlar

    def _assign_parameter_or_buffer(self, fullname: str, tensor: torch.Tensor):
        """

        Diskten okunan tensörü meta modelin icindeki doğru yere yerleştirecek fonksiyon

        Örnek fullname:
            "transformer.wte.weight"

        rpartition(".") son noktadan böler:
            path = "transformer.wte"
            attr = "weight"
        """

        path,_,attr=fullname.rpartition(".")

        module=self.model.get_submodule(path) if path else self.model

        if attr in module._parameters:

            original=module._parameters[attr]
            module._parameters[attr]=nn.Parameter(
                tensor,requires_grad=original.requires_grad
            )
            #bu attr öğrenilebilir bir parametreyse gerçek bir tensörü parameter nesnesine sar requires_grad değerini iskeletteki parameterdan koru


        elif attr in module._buffers:
            #buffer,model  durumunda bulunan ama normalde öğrenilmeyen tensor olabilir tensörü doğrudan değiştir
            module._buffers[attr]=tensor

        else:
            # İsim ne parametre ne de buffer ise checkpoint/model adları uyuşmuyor.
            raise KeyError(f"Modelde bilinmeyen parametre/buffer: {fullname}")



    def _make_pre_hook(self, index: int):
        """
        Verilen katman indeksi için pre-hook fonksiyonu üretir.

        Dış fonksiyon index'i kapatır (closure); böylece her blok kendi
        indeksini bilen bir callback alır
        """
        def hook(module,args):
            #önceki block hala yüklüyse yeni block yüklenmeyecek
            if self._loaded is not None:
                raise RuntimeError(
                    "Önceki Transformer katmanı henüz boşaltılmadı"
                )

            prefix=self.prefixes[index]
            excepted=self.layer_names[index]

            weights=self.store.load_tensors(excepted)# Bu bloğa ait tüm ağırlıkları WeightIndex üzerinden oku.

            nbytes=sum(
                tensor.numel() * tensor.element_size()
                for tensor in weights.values()
            )# Tensorlerin gerçek boyutlarını bayt cinsinden topla.
            # numel() eleman sayısı, element_size() her elemanın bayt boyutudur.

            if (self.max_layer_bytes is not None and nbytes > self.max_layer_bytes):
                raise MemoryError(f"{prefix}: {nbytes} byte > bütçe {self.max_layer_bytes}")


            # Checkpoint anahtarları tam önek içerir:
            # "transformer.h.0.attn.c_attn.weight"
            # Modülün kendi load_state_dict çağrısı ise yalnızca yerel ad bekler:
            # "attn.c_attn.weight"
            local_weights={
                name[len(prefix)+1:]:tensor for name,tensor in weights.items()
            }

            try:
                # Ağırlıkları bu bloğa bağla.
                # assign=True, uygun durumlarda tensorü kopyalamak yerine parametre/buffer referanslarını atamayı sağlar.
                module.load_state_dict(local_weights,strict=True,assign=True)

                self._loaded=index# Bu indeks artık gerçek ağırlıkları taşıyor

                self.layer_reads+=1


            except BaseException:
                # Yükleme sırasında hata çıkarsa bloğu meta durumuna döndür.
                module.to_empty(device="meta")
                raise
        return hook

    def _make_post_hook(self, index: int):
        """verilerin katman indeksi için post-hook üretir, katmanın hesaplaması tamamlanınca gerçek ağırlıkları boşaltır"""
        def hook(module,args,output):
            if self._loaded ==index:#yalnızca şu anda yüklenmiş olan blok kendisiyse boşaltır
                module.to_empty(device="meta")#modül parametrelerini meta aygıtına taşır, gerçek ağırlık verileri artık bu modülün parametrelerinde tutulmaz
                #bu çağrı cpu belleğinin işletim sistemi tarafından tam o anda kesinlikle bırakılacağını garanti etmez
                self._loaded=None
            #forward hook, output döndürürse onu değiştirebilir, aynı outputu döndürerek modelin normal akışını sürdürr

            return output

        return hook

    def resident_layer_count(self)->int:
        """
        kaç transformer bloğunda en az bir gerçek parametre var

        normal bir forward tamamlandıktan sonra sıfır beklenir
        bu metot yalnızca transformer bloklarını sayar embeggin/head gibi kalıcı ağırlıkları ve ara aktivasyonları saymaz
        """
        return sum(any(not parameter.is_meta for parameter in block.parameters()) for block in self.layers)

    @torch.inference_mode()
    def logits(self,input_ids:torch.Tensor)-> torch.Tensor:
        """
        modelin tahmin hesabını başlatan ana metot
        input_ids: [batch_size,sequence_length] biciminde token id tensörü, bu prototip cpu üzerindeki torch.long tensörü bekler
        logits:
            Her konum için sözlükteki tokenlara ait ham puanlar.
            Bunlar tek başına olasılık değildir; softmax uygulanabilir.
        """

        #girdinin tensor olduğundan ve iki boyutlu olduğundan emin olacağız
        if not isinstance(input_ids,torch.Tensor) or input_ids.ndim!=2:
            raise ValueError("input_ids shape [batch, sequence] olmalıdır")

        if input_ids.dtype != torch.long or input_ids.device.type != "cpu":
            raise ValueError(
                "CPU üzerindeki torch.long token ID tensorü bekleniyor"
            )

        #kilit başka bir forward tarafından tutuluyorsa beklemek yerine hata verecek paralel forward desteklemiyoruz şu anda

        if not self._lock.acquire(blocking=False):
            raise RuntimeError(
                "Bu prototip paralel forward işlemlerini desteklemez"
            )

        try:

            # Modelin standart forward fonksiyonunu çalıştır.
            # Transformer bloklarına geldiğinde yukarıda kaydedilen pre/post
            # hook'lar otomatik olarak yükleme ve boşaltma yapar.
            # use_cache=False, geçmiş key/value cache kullanılmasını kapatır.
            return self.model(
                input_ids=input_ids, use_cache=False
            ).logits

        finally:
            #hata olursa bile yüklenmiş bir blok kalırsa temizlemeyi dene
            if self._loaded is not None:
                self.layers[self._loaded].to_empty(device="meta")
                self._loaded=None


            self._lock.release()# Kilidi her durumda bırak; aksi takdirde nesne kilitli kalabilir.

    def close(self):
        """
        bu sınıfın eklediği hookları kaldırır

        Hook'lar kaldırılınca otomatik diskten yükleme/boşaltma davranışı sona erer.
        Bu metot modeli veya dosyaları silmez; yalnızca kayıtlı callback'leri kaldırır.
        """
        for handle in self._handles:
            handle.remove()

        self._handles.clear()