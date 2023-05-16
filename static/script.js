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
function readURL(input) {
    if (input.files && input.files[0]) {
        if (!input.files[0].type.startsWith('image/')) {
            alert("사진을 넣어주세요");
            return;
        }
        var reader = new FileReader();
        reader.onload = function(e) {
            $('.image-title-wrap').hide()
            $('.image-upload-wrap').hide();
            $('#face-image').hide();
            $('#loading').show();
            $('.file-upload-image').attr('src', e.target.result).css('max-height', '300px');
            $('.file-upload-content').show();
            $('.image-title').html(input.files[0].name);
            init().then(function(){
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
$('.image-upload-wrap').bind('dragover', function() {
    $('.image-upload-wrap').addClass('image-dropping');
});
$('.image-upload-wrap').bind('dragleave', function() {
    $('.image-upload-wrap').removeClass('image-dropping');
});

async function init() {
    var image = document.getElementById("face-image");

    // create a Blob object from base64 encoded string
    var byteCharacters = atob(image.src.split(',')[1]);
    var byteNumbers = new Array(byteCharacters.length);
    for (var i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    var byteArray = new Uint8Array(byteNumbers);
    var blob = new Blob([byteArray], { type: 'image/jpeg' });

    const formData = new FormData();
    formData.append("file", blob, "image.jpg");

    const response = await fetch('/photo', {
        method: 'POST',
        body: formData,
    });
    const result = await response.json();
    console.log(result['recommend']);
    console.log(result['predict_arr']);
    var resultmessage;
    switch (result['recommend']){
        case "fat":
            resultmessage='대나무 헬리콥터를 가지고 계신가요?'
            break;
        case "thin":
            resultmessage='진구야?'
            break;
    }
    $('.result-message').html(resultmessage)
    labelContainer = document.getElementById("label-container");
    var barWidth=result['predict_arr']*0.8+"%";

    // 텍스트가 아닌 html 태그를 넣어주고 태그에 막대그래프처럼 크기를 넣어주자
    const newLabel = document.createElement("div");
    var label = "<div class='doraemon-label d-flex align-items-center'>도라에몽 주먹일 확률?</div>"
    var bar = "<div class='bar-container position-relative container'><div class='doraemon-box'></div><div class='d-flex justify-content-center align-items-center doraemon-bar' style='width: " + barWidth + "'><span class='d-block percent-text'>" + result['predict_arr'] + "%</span></div></div>"
    newLabel.innerHTML = label+bar;
    labelContainer.appendChild(newLabel);
}