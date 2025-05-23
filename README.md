<html lang="ja">
    <head>
        <meta charset="utf-8" />
    </head>
    <body>
        <h1><center>VGGT を単一画像深度推定器として使う</center></h1>
        <h2>なにものか？</h2>
        <p>
            VGGTに1枚の画像を入力して深度画像を推定するプログラムです。<br>
            <br>
           ・(project) <a href="https://vgg-t.github.io/">https://vgg-t.github.io/</a><br>
           ・(paper)   <a href="https://arxiv.org/abs/2503.11651">https://arxiv.org/abs/2503.11651</a><br>
           ・(code)     <a href="https://github.com/facebookresearch/vggt">https://github.com/facebookresearch/vggt</a><br>
           ・(demo)    <a href="https://huggingface.co/spaces/facebook/vggt">https://huggingface.co/spaces/facebook/vggt</a><br>
            <br>
            (入力画像)<br>
            <img src="images/input.png"><br>
            (得られた深度画像を使って3D表示)<br>
            <img src="images/result.png">
        </p>
        <h2>環境構築方法</h2>
        <p>
            ● githubからVGGTのコードをダウンロードする<br>
            　<a href="https://github.com/facebookresearch/vggt"?>https://github.com/facebookresearch/vggt</a><br>
            　Code → Download ZIP<br>
            <br>
            ● vggt-main.zip を解凍する<br>
            <br>
            ● 学習済モデルパラメータをダウンロードし vggt-main フォルダに配置する <br>
            　<a href="https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt2">https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt (5.03GB)</a><br>
            　download をクリックする<br>
            <br>
            ● Python 動作環境を構築する<br>
            　・pip install opencv-python<br>
            　・pip install torch torchvision torchaudio <br>
            　　GPUの場合は<a href="https://pytorch.org/">https://pytorch.org/</a> に従って PyTorch 2.xをインストールする<br>
            　・pip install gradio<br>
            　・pip install trimesh<br>
            　・pip install matplotlib<br>
            　・pip install scipy<br>
            　・pip install einops<br>
            　・pip install open3d<br>
        </p>
        <h2>使い方</h2>
        <p>
            ● 画像を入力して深度画像を得る<br>
            　　src/vggt_single_image.py を vggt-main フォルダにコピー<br>
            　　python  vggt_single_image.py (画像ファイルパス)<br>
            <br>
            　・GPUで動作させる場合(未確認)<br>
            　　 vggt_single_image.py の以下2行を変更<br>
            <br>
                　　device = "cpu"<br>
            　　　　↓<br>
                　　device = "cuda" if torch.cuda.is_available() else "cpu"<br>
            <br>
                　　dtype = torch.float32<br>
               　　　↓<br>
                　　dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16<br>
            <br>
                　●RGB画像, 深度画像から3D表示する<br>
                　　python o3d_rgb_depth_to_pcd.py (RGB画像) (深度画像) [(zスケール:省略可)]
        </p>
         <h2>その他</h2>
        <p>
            背景(単色エリア?)が飛び出してくる問題がある。
            <img src="images/fail.gif">
        </p>
        <p>
            この場合、青が飛び出してくるので、blueの輝度が200以上だったらdepthを最大にする(一番奥に引っ込める)<br>
            という場当たり的な対策をすると<br>
            <br>
            import cv2<br>
            TH = 200<br>
            MAX_VALUE = 65535 <br>
            img =cv2.imread("rgb_image.png")<br>
            H, W = img.shape[:2]<br>
            depth = cv2.imread("depth_image.png", cv2.IMREAD_UNCHANGED)<br>
            for y in range(H):<br>
            　for x in range(W):<br>
            　　b = img[y][x][0]<br>
            　　if b > TH:<br>
            　　　depth[y][x] = MAX_VALUE<br>
            cv2.imwrite("modified_depth.png", depth)<br>
        </p>
        <p>
            こんな感じになる。
            <img src="images/modified.gif">
        </p>
    </body>
</html>
