// Captuer result
function captureAndSave() {
    var element = document.body;
    html2canvas(element).then(function(canvas) {
        var dataURL = canvas.toDataURL('image/png');
    
        // Create a temporary anchor element
        var link = document.createElement('a');
        link.href = dataURL;
        link.download = 'capture.png';
    
        // Programmatically click the anchor element to trigger the download
        link.click();
    });
    }
    function clip(){
        var url = '';    // <a>태그에서 호출한 함수인 clip 생성
        var textarea = document.createElement("textarea");  
        //url 변수 생성 후, textarea라는 변수에 textarea의 요소를 생성
    
        document.body.appendChild(textarea); //</body> 바로 위에 textarea를 추가(임시 공간이라 위치는 상관 없음)
        url = window.document.location.href;  //url에는 현재 주소값을 넣어줌
        textarea.value = url;  // textarea 값에 url를 넣어줌
        textarea.select();  //textarea를 설정
        document.execCommand("copy");   // 복사
        document.body.removeChild(textarea); //extarea 요소를 없애줌
    
        alert("URL이 복사되었습니다.")  // 알림창
        }
    
    function resizeImage(image, maxWidth, maxHeight) {
        var canvas = document.createElement('canvas');
        var ctx = canvas.getContext('2d');
    
        var width = image.width;
        var height = image.height;
    
        if (width > height) {
            if (width > maxWidth) {
                height *= maxWidth / width;
                width = maxWidth;
            }
        } else {
            if (height > maxHeight) {
                width *= maxHeight / height;
                height = maxHeight;
            }
        }
    
        canvas.width = width;
        canvas.height = height;
    
        ctx.drawImage(image, 0, 0, width, height);
    
        return new Promise((resolve, reject) => {
            canvas.toBlob(function (blob) {
                resolve(blob);
            }, 'image/jpeg', 0.7); // 리사이즈된 이미지를 Blob 형식으로 압축하여 0.7 품질로 저장
        });
    }
        
    
    let result;
    
    function base64ToByteArray(base64) {
        try {
          const binaryString = atob(base64);
          const byteArray = new Uint8Array(binaryString.length);
          for (let i = 0; i < binaryString.length; i++) {
            byteArray[i] = binaryString.charCodeAt(i);
          }
          return byteArray;
        } catch (error) {
          console.error('Failed to decode base64:', error);
          return null;
        }
      }
    
      async function readURL(input) {
        if (input.files && input.files[0]) {
            if (!input.files[0].type.startsWith('image/')) {
                alert("");
                return;
            }
            var reader = new FileReader();
            reader.onload = async function (e) {
                $('.file-upload-content').show();
                $('#loading').show();
                $('.image-title-wrap').hide();
                $('.image-upload-wrap').hide();
                $('#face-image').hide();
                var base64String = e.target.result.split(',')[1];
    
                var image = new Image();
                image.onload = async function () {
                    var blob = await resizeImage(image, 500, 500); // 이미지 크기를 500x500 픽셀로 리사이즈
    
                    const formData = new FormData();
                    formData.append('file', blob, 'image.jpg');
    
                    try {
                        const response = await fetch('/whos_the_king', {
                            method: 'POST',
                            body: formData,
                        });
    
                        result = await response.json();
    
                        const dataUrl = `data:image/jpeg;base64,${result['image']}`;
    
                        $('.file-upload-image').attr('src', dataUrl).css('max-height', '350px');
                        $('.image-title').html(input.files[0].name);
                        init().then(function () {
                            $('#loading').hide();
                            $('.image-title-wrap').show()
                            $('#face-image').show();
                            document.querySelector(".sns_wrap").style.display = "flex";
                        });
                    } catch (error) {
                        console.error(error);
                    }
                };
                image.src = e.target.result;
            };
            reader.readAsDataURL(input.files[0]);
        } else {
            removeUpload();
        }
    }
    
    
    function removeUpload() {
        $('.file-upload-input').replaceWith($('.file-upload-input').clone());
        $('.file-upload-content').hide();
        $('.image-upload-wrap').show();
    }
    $('.image-upload-wrap').bind('dragover', function() {
        $('.image-upload-wrap').addClass('image-dropping');
    });
    $('.image-upload-wrap').bind('dragleave', function() {
        $('.image-upload-wrap').removeClass('image-dropping');
    });
    
    async function init() {
        var resultmessage;
        resultmessage = result['message'];
        $('.result-message').html(resultmessage);
    }
    
    