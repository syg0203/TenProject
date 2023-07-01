function captureAndSave() {
    var element = document.body;
    html2canvas(element).then(function (canvas) {
        var dataURL = canvas.toDataURL('image/png');

        // Create a temporary anchor element
        var link = document.createElement('a');
        link.href = dataURL;
        link.download = 'capture.png';

        // Programmatically click the anchor element to trigger the download
        link.click();
    });
}
function clip() {
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

function readURL(input) {
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
            var base65String = e.target.result.split(',')[1]

            var byteCharacters = base64ToByteArray(base65String);
            var byteArray = new Uint8Array(byteCharacters);
            var blob = new Blob([byteArray], { type: 'image/jpeg' });
            const formData = new FormData();
            formData.append("file", blob, "image.jpg");

            const response = await fetch('/photo', {
                method: 'POST',
                body: formData,
            });

            result = await response.json();
            console.log(result)

            const dataUrl = `data:image/jpeg;base64,${result['filename']}`;

            $('.file-upload-image').attr('src', dataUrl).css('max-height', '300px');
            $('.image-title').html(input.files[0].name);
            init().then(function () {
                $('#loading').hide();
                $('.image-title-wrap').show()
                $('#face-image').show();
                document.querySelector(".sns_wrap").style.display = "flex";
            });
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
$('.image-upload-wrap').bind('dragover', function () {
    $('.image-upload-wrap').addClass('image-dropping');
});
$('.image-upload-wrap').bind('dragleave', function () {
    $('.image-upload-wrap').removeClass('image-dropping');
});

async function init() {
    var resultmessage;
    if (result['predict_arr'] >= 90) {
        resultmessage = '도라에몽?'
    } else if (result['predict_arr'] < 90 && result['predict_arr'] >= 80) {
        resultmessage = '대나무 헬리콥터를 가지고 계신가요?'
    } else if (result['predict_arr'] < 80 && result['predict_arr'] >= 50) {
        resultmessage = '도라에몽 위험'
    } else if (result['predict_arr'] < 50 && result['predict_arr'] >= 1) {
        resultmessage = '진구?'
    } else if (result['predict_arr'] == 0) {
        resultmessage = '주먹이 탐지되지 않았습니다.'
    }
    $('.result-message').html(resultmessage)
    labelContainer = document.getElementById("label-container");
    var barWidth = result['predict_arr'] * 0.8 + "%";

    // 텍스트가 아닌 html 태그를 넣어주고 태그에 막대그래프처럼 크기를 넣어주자
    const newLabel = document.createElement("div");
    var label = "<div class='doraemon-label d-flex align-items-center'>도라에몽 주먹일 확률?</div>"
    var bar = "<div class='bar-container position-relative container'><div class='doraemon-box'></div><div class='d-flex justify-content-center align-items-center doraemon-bar' style='width: " + barWidth + "'><span class='d-block percent-text'>" + result['predict_arr'] + "%</span></div></div>"
    newLabel.innerHTML = label + bar;
    labelContainer.appendChild(newLabel);
}