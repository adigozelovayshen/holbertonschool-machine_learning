# Bayesian Probability

Bu layihə **Bayesian Probability** (Beyz Ehtimalı) mövzusuna dair fundamental anlayışları və onların Python (NumPy) vasitəsilə tətbiqini əhatə edir. Layihə çərçivəsində bir xərçəng dərmanının yan təsirlərinin binomial paylanma əsasında analizi aparılır.

---

## Layihənin Məqsədi

Layihənin əsas məqsədi verilmiş müşahidə məlumatları ($ xəstə sayı, $ yan təsir göstərən xəstə sayı) əsasında müxtəlif ehtimal fərziyyələrini qiymətləndirməkdir. 

### Əhatə olunan anlayışlar:
* **Likelihood:** Müşahidə olunan məlumatın müxtəlif $ ehtimalları altında baş vermə dərəcəsi.
* **Prior Probability:** Heç bir məlumat olmadan əvvəlki ilkin inanc/ehtimal.
* **Posterior Probability:** Yeni məlumat daxil olduqdan sonra yenilənmiş ehtimal.
* **Marginal Probability (Evidence):** Məlumatın baş verməsinin ümumi ehtimalı.

---

## Riyazi Düsturlar

### 1. Binomial Likelihood
Müşahidə olunan $ uğur sayının $ sınaq daxilində ehtimalı:
$$P(x | n, p) = \binom{n}{x} p^x (1 - p)^{n-x}$$

### 2. Bayes Teoremi
$$P(A | B) = \frac{P(B | A) P(A)}{P(B)}$$

Burada:
- **P(A | B):** Posterior
- **P(B | A):** Likelihood
- **P(A):** Prior
- **P(B):** Marginal Probability (Evidence)

---

## Fayl Strukturu

| Fayl | Təsvir |
| --- | --- |
| `0-likelihood.py` | Binomial paylanma üçün likelihood dəyərlərini hesablayan funksiya. |
| `0-main.py` | Likelihood funksiyasını test etmək üçün skript. |

---

## Quraşdırma və İstifadə

### Tələblər
- Python 3.10+
- NumPy

### Quraşdırma
```bash
pip install numpy
```

### İcra
Hər hansı bir taskı yoxlamaq üçün:
```bash
python3 0-main.py
```

---

## Müəllif
**Ayshen Adigozelova**
*Holberton School - Machine Learning Student*
