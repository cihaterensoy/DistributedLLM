"""
farklı transformer mimarilerinin katman yapılarını standart bir arayüzle bulmamızı sağlayna modül
"""

from __future__ import annotations # Tür açıklamalarının değerlendirilmesini ertelenmiş hâle getirir.
from dataclasses import dataclass  #Veri taşıyan sınıfları kısa yazmamızı sağlar
from typing import Any

@dataclass(frozen=True)
class ModelAdapter:
    """
    belirli bir model türü için path tutan veri sınıfı
    frozen=true ise Nesne oluşturulduktan sonra alanlarının değiştirilmesini engeller, böylece adapter yanlışlıkla başka bir mimariye dönüştürülemez
    """
    model_type:str # örneğin gpt2 llama qwen2
    layer_path:str #model nesnesinn içindeki katman yolu örneğin transofrmer.h

    def layers(self, model:Any):
        """
        verilen model nesnesinden transformer bloklarının koleksiyonunun döndürür
        """
        current =model

        for piece in self.layer_path.split("."):

            current = getattr(current, piece)
            #önce model.transformer ardından transformer.h nesnesine erişiriz yol yanlışsa getattr error verecek

        #transformer blockları genellikle pytorch modulelist bicimindedir
        if not hasattr(current,"__len__"):
            raise TypeError(f"{self.layer_path} geçerli bir katman koleksiyonu değil")
        #bu kontrol tek başına nesnenin kesinlikle doğru block koleksiyonu olduğunu garanti etmez sadece basit bir doğrulamadır

        return current #transformer blocklarını tutan modulelist

    def prefix(self,index:int)->str:
        """
        belirli bir katmanın state_dict içindeki önekini üretir
        Bu önek daha sonra o bloğa ait ağırlıkları seçmek için kullanılır
        :param index:
        :return:
        """

        if index < 0:
            raise ValueError("Katman indeksi negatif olamaz")
        return f"{self.layer_path}.{index}" # Örneğin "transformer.h" + "." + "3"


_ADAPTERS = {
    "gpt2": ModelAdapter("gpt2", "transformer.h"),
    "llama": ModelAdapter("llama", "model.layers"),
    "qwen2": ModelAdapter("qwen2", "model.layers"),
}

def get_adapter(model_type:str)->ModelAdapter:
    """
    Model türü adına göre doğru adaptörü döndürür tanınmayan için valueerror verecek
    :param model_type:
    :return:
    """
    try:
        return _ADAPTERS[model_type]
    except KeyError as error:
        supported = ", ".join(sorted(_ADAPTERS))

        raise ValueError(
            f"Desteklenmeyen model_type={model_type!r}. "
            f"Desteklenenler: {supported}"
        ) from error  # Asıl KeyError'ı hata zincirinde koru.