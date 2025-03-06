# Welcome To Warga Slowy

Terima kasih sudah ingin berkontribusi di repository `WargaSlowy`, sebelum kamu melakukan pull request usahakan kamu mengikuti aturan dibawah ini
- kode kamu harus original dari milik kamu, jika milik orang lain, berikan pengecualian dan berikan informasi serta referensinya
- ketika kode kamu sudah di pull request maka kode kamu sudah berlisensi [MIT](LICENSE)
- kamu bisa menambahkan informasi pada `docstring`, atau menambahkan `komentar` untuk menambah kejelasan dari kode

## Cara lakukan pull request

untuk melakukan pull request kamu bisa mengikuti langkah dibawah ini:

- pastikan kamu fork terlebih dahulu repositori ini
- kemudian lakukan clone dari repository yang kamu fork tadi
    ```bash
    git clone https://github.com/nama_user/transformer
    ```
    informasi: ``nama_user`` -> nama user github kamu
- setelah di clone pastikan kamu membuat branch baru
    ```
    git checkout -b nama_branch_yang_kamu_buat
    ```

jika kamu menggunakan ``uv``
- pastikan kamu install terlebih dahulu package yang terdapat pada uv
- kemudian kamu bisa jalankan perintah
    ```
    uv run pytest --verbose
    ```

jika kamu tidak menggunakan ``uv``
- pastikan kamu install depdencies atau library yang digunakan untuk membangun project ini
    ```
    torch
    # untuk unittesting
    pytest
    ```

setelah kamu sudah melakukan perubahan kemudian kamu bisa menabahkan hasil perubahan kamu dengan cara
```
git add .
```
kemudian commit dengan mengikuti standar dari commit konvensional, kamu bisa lihat [disini](https://www.conventionalcommits.org/en/v1.0.0/), sebagai contoh
```
git commit -m "docs: menambahkan dokumentasi kode" -m "- menambhakan iformasi input pada blok transformer"
```
kemudian lakukan push
```
git push origin nama_branch_yang_kamu_buat
```
