"""
GPT2'nin transformer bloklarını sabit diskten sırayla yükleyen cput motoru
"""

from __future__ import annotations
import json
from fileinput import filename
from pathlib import Path
from typing import Any
import torch
from safetensors.torch import load_file, save_file

_FORMAT_VERSION = 1

def _save_tensors(path:Path, tensors:dict[str, torch.Tensor]) -> None:
    #tensörleri safetensors formatında diske kaydeder
    save_file({k:v.detach().to(device="cpu").contiguous().clone() for k,v in tensors.items()},str(path))
    #detach -> gradyan grafiğinden ayırır
    #contiguous() - Bellekte ardışık (kesintisiz) hale getirir (safetensors için zorunludur)
    #clone() Bellekte paylaşılan depolama (storage sharing) çakışmalarını önler

def export_gpt2_shards(model:Any,folder: str | Path) -> Path:
    #1)bellekteki modeli tek seferliğine katman dosyalarına bölüp diske kaydedyoruz

    if getattr(model.config,"model_type",None)!="gpt2":
        raise ValueError("model_type='gpt2' destekler")
    if model.training: # Eğitim modundaki modellerde dropout gibi katmanlar farklı davranabilir bu yüzden model.eval'i çağrıyorum önce
        raise ValueError("Önce model.eval() çağırın")
    if model.transformer.wte.weight.device.type=="meta": #Meta tensör, gerçek ağırlık verisini bellekte tutmaz bu üyzden meta cihazındaki bir modelden gerçek ağırlıkları dışarı aktaramayız
        raise ValueError("Meta modelden ağırlık dışarı aktarılamaz")

    output=Path(folder)
    output.mkdir(parents=True,exist_ok=True)
    transformer = model.transformer#gpt'nin transformer bölümünü isimlendirdik

    #ağırlıkları paylaşan katmanları kontrol edeceğimiz kısım
    #gpt2'de toen embedding katmanı ile cıkış katmanı genellikle aynı ağırlık matrisini kullanır
    tied = (model.lm_head.weight.data_ptr() == transformer.wte.weight.data_ptr())

    #2)sürekli gereken ağırlıkları kaydetme
    #bu ağırlıklar her metin işleminde kullanılıyo

    io_tensors={ #her ağırlığın adını ve karşılık gelen tensörü birlikte tutar
        "wte.weight":transformer.wte.weight,#Token embedding ağırlıkları
        "wpe.weight":transformer.wpe.weight,#pozisyon embedding ağırlıkları tokenların metin içindeki konumlarını temsil ediyo
        "ln_f.weight":transformer.ln_f.weight,#son layernorm katmanının ölcekleme ve kaydırma parametreleri
        "ln_f.bias": transformer.ln_f.bias,
    }

    #eğer lm_head ve wte aynı ağırlıkları paylaşmıyorsa cıkış katmanının ağırlıklarınıda dosyaya kaydetmeliyixz
    #yoksa bu ağırıklar meta modelden geri getirilemez

    if not tied:
        io_tensors["lm_head.weight"]=model.lm_head.weight

    # hazırladığımız tensörleri io.safetensors dosyasına yazıyoruz.
    _save_tensors(output / "io.safetensors", io_tensors)

    #3) her transformer bloğunu ayrı kaydetme
    sizes: list[int]=[] #her bloğun dosya boyutunu kaydedeceğimiz liste

    for idx,block in enumerate(transformer.h):#gpt2'nin transformer bloklarını ve index numarasını aldım
        filename = f"layer_{idx:04d}.safetensors"# her bloğa farklı isim verdim

        _save_tensors(output / filename, block.state_dict())
        #bloğun mevcut öğrenilmiş parametrelerini ve kayıtlı durumunu ismiyle birlikte sözlük halinde almamıza yarar state_dict

        #dosya diske yazıldıktan sonra gerçek boyutunu bayt cinsinden öğrenip listeye ekliyoruz
        #daha sonra bu bilgi, her katman dosyasının belirlenen dosya boyutunu sınırını aşıp aşmadığını kontrol etmek için kullanacağım

        sizes.append((output/filename).stat().st_size)

    #4)modelin yapılandırma bilgisini kaydetme

    model.config.save_pretrained(output)
    #5)manifest dosyasını oluşturma
    manifest = {
        "format_version": _FORMAT_VERSION,#Dosya düzeninin sürümü
        "architecture": "gpt2",#Kullanılan model mimarisi
        "num_layers": len(transformer.h),#Toplam Transformer bloğu sayısı
        "tied_word_embeddings": tied,#Giriş embedding'i ile çıkış katmanının ağırlıkları paylaşılıyor mu?
        "layer_files": [
            f"layer_{i:04d}.safetensors"
            for i in range(len(transformer.h))
        ],#Katman dosyalarının sıralı listesi
        "layer_file_bytes": sizes,#Her katman dosyasının boyutu
    }

    (output/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    #indent=2 JSON dosyasını okunması kolay olacak şekilde biçimlendirir
    return output

class DiskOffloadedGPT2:
    """
        GPT-2'yi katman katman diskten yükleyerek çalıştıran sınıf.

        Sınıfın görevleri:
        1. Daha önce oluşturulan model dosyalarını okumak.
        2. Modelin iskeletini gerçek ağırlıklar ayırmadan oluşturmak.
        3. Sürekli gereken embedding ve LayerNorm ağırlıklarını RAM'e yüklemek.
        4. İstenilen metni Transformer katmanlarından sırayla geçirmek.
        5. Her katmanın ağırlıklarını iş bitince bellekten çıkarmak.
        """

    def __init__(self,folder:str|Path,max_layer_file_bytes:int | None=None):
        #folder - export_gpt2_shards() tarafından oluşturulmuş klasör
        #max_layer_.. bir katman dosyasının izin verilen en büyük boyutu
        from transformers import GPT2Config, GPT2LMHeadModel
        self.folder=Path(folder)
        #manifest dosyasının json içeriğini alıyoruz
        manifest = json.loads(
            (self.folder / "manifest.json").read_text(
                encoding="utf-8"
            )
        )

        #1)dosya formatını ve model mimarisini doğrulama
        if(manifest.get("format_version")!=_FORMAT_VERSION or manifest.get("architecture")!="gpt2"):
            raise ValueError("Desteklenmeyen shard formatı")

        config =GPT2Config.from_pretrained(str(self.folder),local_files_only=True) #gpt2'nin config dosyasını okuyorz

        if(config.model_type!="gpt2" or int(config.n_layer)!=manifest["num_layers"]):
            raise ValueError("Config ve manifest uyumsuz")

        expected_files = [ f"layer_{i:04d}.safetensors" for i in range(config.n_layer)] #beklediğim,iz katman dosyalarının isimlerini ve sıralamsını yeniden oluşturuyoruzx

        #manfiest içindeki dosya listesi beklenen listeyle aynı olmalı böylece yanlış sıralanmış veya eksik bir listesyi tespit ediyoruz
        if manifest["layer_files"] != expected_files:
            raise ValueError(
                "Manifest layer listesi beklenen sıralamayla uyuşmuyor"
            )
        #bu kontrol dosyaların gerçekten mevcut veya bozulmamış olduğunu tek başına garanti etmiyor

        #2)katman dosaysı boyutu sınırını kontrol etme

        #kullanıcı bir maksimum dosya belirtmişse her katmanı bu sınıra göre kotnrol edeceğiz

        if max_layer_file_bytes is not None:
            if max_layer_file_bytes <=0:
                raise ValueError(
                    "max_layer_file_bytes pozitif olmalı"
                )

            if any((self.folder/filename).stat().st_size>max_layer_file_bytes for filename in expected_files):
                raise MemoryError(
                    "En az bir katman dosyası ayarlanan bütçeden büyük"
                )
            #her dosyanın gerçek boyutunu diskten okuduk, herhangi bir dosya sınırdan büyükse hata oluştur

        #3)modelin ağırlıksız iskeletini oluşturma
        with torch.device("meta"):
            #model yapısını yalnızca meta tensörlerle oluşturuyoruz
            #gerçek ağırlıklar daha sonra dosyadan yerleştireceğiz
            model = GPT2LMHeadModel(config)
        #normalde gpt2lmheadmodel(config) cağrıldığında modelin parametreleri için gerçek bellek ayrılır
        #burada torch.device("meta") kullanıyoruz - modelin katmanlarını ve tensör şekillerini tanımlar, gerçek ağırlık değerlerini saklamaz, böylece modelin bütün katman ağırlıklarını ram'e ayırmayız
        #python nesneleri, model yapısı ve diğer çalışma verilerini yine bellekte kullanır

        model.eval()#değerlendirme moduna aldık

        self.model = model #modeli sınıftan erişilebilir hale getirdik
        self.manifest = manifest

        #4)sürekli gereken ağırlıkları ram'e yükleme
        #io.safetensors dosyasını cu belleğine yüklüyoruz

        io=load_file(
            str(self.folder / "io.safetensors"),
            device="cpu",
        )
        #giriş embeddiğng katmanlarını gerçek ağırlıklarla dolduruyoruz
        for key,module in (("wte.weight",model.transformer.wte),("wpe.weight",model.transformer.wpe)):
            _,result=key.split(".",1)
            #sonucu ["wte","weight"]
            #result'ımız weight olacak
            #cünkü module.load_state_dict() çağrısında doğrudan embedding katmanını verdiğimiz için anahtarın wte.weight değil yalnızca weight olması gerekir

            module.load_state_dict({result:io[key]},assign=True,strict=True)
            #bir katmanın parametrelerini state_dict içersinden yükler assign-> meta parametreye değerleir kopyalamak yerine verilen tensörleri parametre olarak atar
            #strict =true beklenen anahtarlarla verilen anahtarların tam olarak uyuşmasını ister; eksik veya fazladan anahtarlarda hata verir.
            #böylece meta ağırlıkların yerine gerçek cpu tensörleri gelir

        #son layerNorm ağırlıklarını
        model.transformer.ln_f.load_state_dict({
            "weight":io["ln_f.weight"],
            "bias":io["ln_f.bias"]
        },assign=True,strict=True)

        #5)giriş ve çıkış ağırlıklarını eşleştirme

        #eğer token embedding ve lm_head aynı ağırlıkları kullanıyorsa cıkış katmanını yüjklemeye ayrıca gerek yok
        if manifest["tied_word_embeddings"]:
            model.lm_head.weight=model.transformer.wte.weight
        else:
            model.lm_head.load_state_dict(
                {"weight":io["lm_head.weight"]},
                assign=True,strict=True
            )
            #ağırlıklar paylaşılmıyor sa lm_head için ayrı bir tensör vardır bu tensörü io dosyasınından cıkış katmanına yüklüyoruz


    def resident_layer_count(self)->int:
        #gerçek ağırlıkları ram'de bulunan transformer bloklarının sayısını verir bu sayede katmanların yanlışlıkla bellekte kalıp kalmadığını kontrol ederiz

        #p.is_meta - parametrenin meta cihazında olup olmadığıını söyler
        #any sayesinde parametrelerin en az biri gerçek tensöre true döndürür
        #böylece en az bir gerçek parametresi olan kaç blok bulunduğunu saymış oluruz
        return sum(
            any(not p.is_meta for p in block.parameters())
            for block in self.model.transformer.h
        )
    def logits(self,input_ids:torch.Tensor)->torch.Tensor:
        """
        Girdi tokenlarından modelin logits çıktısını üretir.

        input_ids:
            Şekli [batch, sequence] olan CPU üzerindeki torch.long tensörü.
            batch:
                Aynı anda işlenen metin örneği sayısı.
            sequence:
                Her metin örneğindeki token sayısı.
        logits:
            Her konum için sözlükteki olası tokenlara verilen ham puanlar.
            Örneğin çıktı şekli:
            [batch, sequence, vocab_size]
            Burada vocab_size, modelin bildiği token sayısıdır.
        Bu fonksiyon yeni token üretmez.
        Token üretmek için logits değerleri ayrıca değerlendirilmelidir.
        """
        #1)girdiyi doğrulama
        #girdi pytorch tensörü ve iki boyutlu olması gerekiyor örneğin batch,sequence [[15496, 995, 0]] batch=1 sequence=3
        if(not isinstance(input_ids,torch.Tensor) or input_ids.ndim!=2):
            raise ValueError("input_ids must have shape [batch, sequence]")

        #input.ids.shape iki değerden oluşmalıdır
        batch,length=input_ids.shape

        if(batch<=0 or length<=0 or length>self.model.config.n_positions):
            raise ValueError(
                "Geçersiz batch / sequence uzunluğu"
            )

        #2)gradyan hesaplanmasının kapalı olduğunu kontrol etme
        if torch.is_grad_enabled():
            raise RuntimeError(
                "Bu motor torch.inference_mode() altında çağrılmalı"
            )
        #önceki bir işelmden transformer arğılıkları ramde kalmış mı
        if self.resident_layer_count() !=0:
            raise RuntimeError(
                "Çalıştırma öncesinde RAM'de blok ağırlıkları kalmış"
            )

        tr =self.model.transformer

        #3)token ve pozisyon embedding hesaplaması
        #tr.wte(inputs_ids) her token kimlşiğini öğrenilmiş bir vektöre dönüştürecek
        # gird i [batch, sequence] çıktı [batch, sequence, n_embd] n_embd, her tokenın temsil edildiği vektörün boyutudur
        #tr.wpe(position) her tokenın bulunduğu konuma ait vektörü üretir

        position = torch.arange(
            length,
            dtype=torch.long
        ).unsqueeze(0)
        #birinci boyuta batch boyutunu ekledik

        hidden = tr.drop(tr.wte(input_ids)+tr.wpe(position))
        #token embedding ve pozisyon embedding değelrerini topluyoruz
        #aynı konumdaki token ve pozisyon vektörlerinin boyutlarıda aynıdır
        #tr.drop-> gp2'nin embedding dropout katmanıdır, eval() modunda dropout rastgele elemean kapatmıyor
        # hidden Bir sonraki Transformer bloğuna gönderilecek gizli temsil

        #4) transformer blocklarını tek tek işleme
        for i,block in enumerate(tr.h):
            #1) mevcut bloğun ağırlığını diskten yükeleme
            #manifest içindeki i.dosyanın yolunu oluşturuyoruz
            #döenen weights bir sözlüktür içinde ilgili transformer bloğunun gerçek ağırlıkları bulunur
            weights=load_file(
                str(
                    self.folder/self.manifest["layer_files"][i]
                ),
                device="cpu",
            )
            try:
                #adım2 ağırlıları bloğa yerleştirem
                block.load_state_dict(
                    weights,
                    assign=True,
                    strict=True
                )
                #ağırlık sözlüğüne ihtiyacımız kalmadı cünkü yükleme sonrasında tesnörler blok parametremlenrine atanmıştır
                del weights

                #3)bloğu çalıştırma
                result=block(
                    hidden,
                    use_cache=False
                )
                #hidden tensörünü mevuct ttransformer blockuna çeviriyoruz
                # Gizli temsili bloğun attention ve MLP işlemlerinden geçirir
                #use_cache=False geçmiş tokenlara ait key/value önbelleğini üretmez bu kodda her çağrı verilen diziyi doğrudan işlemek üzere tasarlanmıştır

                #gpt2nin bazı sürümlerinde blok cıktısı birden fazla eleman içeren bir tuple olabilri

                next_hidden=(
                    result[0]
                    if isinstance(result,(tuple,list))
                    else result
                )
                #bazı sürümlerde cıktı doğrudan tensör olabilir
                #bazılarında ise ilk eleman temsil tensörüdür
                #tuple veya list ise ilk elemanı aldık değilse resultın kendisini kullandık

                if(not isinstance(next_hidden,torch.Tensor) or next_hidden.shape != (batch,length,self.model.config.n_embd)):
                    raise RuntimeError(f"Layer {i}: beklenmeyen çıktının boyutu")
                hidden=next_hidden
                #hidden artık bir sonraki katmanın kuullanacağı sonuçtur
                del result,next_hidden
                #ara değişkenlerin refaranslarını kaldırıyoruz,hidden kendi referansını koruduğu için hesaplaam sonucu bir sonraki katmana aktarılamya devam eder
            finally:
                block.to_empty(device="meta")
                #finally hesaplamada bir hata olsa bile calışacak
                #bloğun parametrelerini meta cihaza taşır
                #böylece mevcut bloğun gerçek ağırlıkları model içinde tutulmaya devam etmez
                #hidden tensörü ayrı bir cpu tensörüdür bu işlem hiddeni silmez
                # Sonuç,mevcut bloğun ağırlıkları bellekten çıkarılır gizli temsil sonraki blok için korunur. döngü sonraki katmana geçer
            #5)son normalizasyon ve logits hesaplaması

        #bütün transformer blokları tamamlandı
        #artık hidden, son transformer bloğunun cıktısını içerir
        hidden=tr.ln_f(hidden)
        #ln_f transformer bloklarından sonra yer alan layer norm, ağırlıklar başlangıçta rame yüklenmişti

        #lm_head sözlükteki her tokenı ham puana dönüştürecek

        return self.model.lm_head(hidden)

