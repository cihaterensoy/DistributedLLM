"""
model ağırlıkalrını hangi safetensors dosyasında bulunduğunu takip etmek ve istenen tensörleri okuma yapılacak

iki farklı senaryo desteklenecek
birden fazla shard dosyası ve model.safetensors.index.json
tek bir .safetensors dosyası
"""

from __future__ import annotations
import json
from pathlib import Path
from safetensors import safe_open# Safetensors içindeki tensörleri açmak için.
import torch

class WeightIndex:
    def __init__(self,model_dir: str | Path):
        #model klasörünü inceler ve tensör adı dosya adı yol haritası oluşturur

        self.folder=Path(model_dir).resolve()#gelen klasör yolunu ath nesnesine çevir

        if not self.folder.is_dir():
            raise FileNotFoundError(f"{model_dir} is not a directory")
        index_file=self.folder/"model.safetensors.index.json"# Hugging Face'in çok parçalı safetensors modellerinde genellikle ağırlık adlarını dosyalara eşleyen bir index JSON bulunur,

        if index_file.exists():
            index_data=json.loads(index_file.read_text(encoding="utf-8"))
            mapping = index_data.get("weight_map")

            if not isinstance(mapping, dict) or not mapping: # Dosyada kullanılabilir, boş olmayan bir sözlük bulunmalı
                raise ValueError("Geçersiz safetensors index.json")

            self.mapping=dict(mapping)# Haritanın kopyasını sınıf alanında sakla
        else:
            #index yoksa klasördeki .safetensors dosyalarını bul
            candidates=sorted(self.folder.glob("*.safetensors"))

            if not candidates:
                raise FileNotFoundError(
                    "model.safetensors veya indexli safetensors shardları yok"
                )

            if len(candidates)>1:
                raise ValueError(
                    "Birden fazla .safetensors dosyası için index.json gereklidir"
                )

            #tek dosyayas safe_open dosyanın tensör anahtarlarını listeler tensor değerlerini get_tensor ile almayız
            with safe_open(
                str(candidates[0]),framework="pt",device="cpu"

            ) as source:
                self.mapping={
                    # Her tensör adını o tek dosyanın adıyla eşleştir
                    name:candidates[0].name for name in source.keys()
                }

        #indeks içersindeki dosya adlarını doğrulayacağız

        for filename in set(self.mapping.values()):
            if not isinstance(filename,str) or not filename.endswith(".safetensors"):
                raise ValueError("Yalnızca safetensors dosyaları kabul edilir")

            path = (self.folder/filename).resolve()
            if path.parent != self.folder or not path.is_file():
                raise ValueError(f"Geçersiz veya eksik shard dosyası: {filename}")
            # path.parent == self.folder kontrolü dosyanın doğrudan bu klasörde
            # olmasını şart koşar. is_file() ise dosyanın gerçekten varlığını denetler.

    def keys(self)->set[str]:
        """
        modeldeki tensör isimlerini küme olarak döndürür {"transformer.wte.weight", "transformer.wpe.weight", ...} tarzında
        :return:
        """
        return set(self.mapping)

    def load_tensors(self,names:list[str])->dict[str, torch.Tensor]:

        """
        istenen tensörleri dosyalarından okuyup sözlük olarak kaydeder
        Girdi:
            ["model.layers.0.input_layernorm.weight", ...]

        Çıktı:
            {"model.layers.0.input_layernorm.weight": tensor(...), ...}

        bu metot bütün modeli okumak yerine yalnızca names listesinde istenen tensörleri almaya çalışır ve bunlar ramde yer tutar
        :param names:
        :return:
        """

        missing = set(names)-self.keys() #istenen isimlerden model haritasında olmayanları bul
        if missing:
            raise KeyError(f"Model ağırlıkları eksik: {sorted(missing)[:5]}")

        #istenen tensörler bulunduysa bulundukları dosyaya göre gruplayacağız böylece aynı dosyayı her tensör için tekrar açmak zorunda kalamyız

        grouped: dict[str,list[str]]={}
        for name in names:
            filename = self.mapping[name]
            grouped.setdefault(filename, []).append(name)

        result: dict[str, torch.Tensor] = {}

        for filename,group in grouped.items():
            with safe_open(
                str(self.folder/filename),framework="pt",device="cpu"
            )as source:
                for name in group:
                    # İstenen tensorü CPU üzerinde alıp sonuç sözlüğüne koy.
                    # Tensör büyükse bu tensörün kendisi RAM'de önemli yer tutar.
                    result[name]=source.get_tensor(name)

        return result # Çağıran kod bu tensörleri model katmanına yerleştirebilir




