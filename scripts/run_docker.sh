sudo docker run -d --gpus all --shm-size=240g -p 6006:6006 --name lid_container \
-v "$(pwd)/../Large-Image-Diffusion:/app/Large-Image-Diffusion" \
-v /usr/local/share/ca-certificates/verdi.crt:/usr/local/share/ca-certificates/verdi.crt \
-e HTTP_PROXY=https://10.253.254.250:3130/ \
-e HTTPS_PROXY=https://10.253.254.250:3130/ \
-e http_proxy=https://10.253.254.250:3130/ \
-e https_proxy=https://10.253.254.250:3130/ \
-e REQUESTS_CA_BUNDLE=/usr/local/share/ca-certificates/verdi.crt \
hale0007/lid:1.0.1 tail -f /dev/null