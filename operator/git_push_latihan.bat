@echo off
cd /d D:\PYTHON\Sesi1

:: Inisialisasi git kalau belum ada .git
if not exist ".git" (
    git init
    echo ✅ Git repository initialized
)

:: Pastikan branch utama bernama main
git branch -M main

:: Cek apakah remote origin sudah ada
git remote | findstr "origin" >nul
if %errorlevel%==0 (
    echo ℹ️ Remote 'origin' sudah ada, update URL...
    git remote set-url origin https://github.com/affiers/Pertemuan-1
) else (
    echo ➕ Menambahkan remote 'origin'
    git remote add origin https://github.com/affiers/Pertemuan-1
)

:: Tambahkan semua file
git add .

:: Tanya pesan commit ke user
set /p commit_msg=Masukkan pesan commit: 

:: Commit perubahan (jika ada)
git commit -m "%commit_msg%" || echo ⚠️ Tidak ada perubahan untuk di-commit

:: Push ke GitHub
git push -u origin main

pause
