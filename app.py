import tempfile
import streamlit as st
from ultralytics import YOLO
from predict import video_stream, image_stream
from PIL import Image
model = YOLO('best.pt')



def main():
    st.title('Object Detection with YOLO-8')
    st.sidebar.title('Settings')
    st.sidebar.subheader('parameters')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 300px;
        }
        [data-testid="stSidebar"][aria-expanded="False"] > div:first-child {
        width: 300px;
        margin-left: -300px;
        }
        </style>
        """,
        unsafe_allow_html=True,

    )
    app_mode = st.sidebar.selectbox(
        'Choose the App Mode',
        ['About the App', 'Run on Image', 'Run on Video']
    )

    if app_mode == 'About the App':
        st.markdown('This project build on YOLO-8 State of Art model')
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 300px;
            }
            [data-testid="stSidebar"][aria-expanded="False"] > div:first-child {
            width: 300px;
            margin-left: -300px;
            }
            </style>
            """,
            unsafe_allow_html=True,

        )
        st.video("https://www.youtube.com/watch?v=YSgm0DQ9F9g&list=PLUE9cBml08yiahlgN1BDv_8dAJFeCIR1u")
        st.markdown("""
        Object Detection project done by gayathri
        """)

    elif app_mode == 'Run on Image':
        image_file = st.sidebar.file_uploader('Upload an Image', type=['jpg', 'jpeg', 'png'])
        demo_image = 'demo_image.jpeg'
        if image_file is not None:
            uploaded_image = Image.open(image_file)
            st.sidebar.text('Uploaded Image')
            st.sidebar.image(image_file)
            processed_images = image_stream(uploaded_image)
            for idx, processed_image in enumerate(processed_images):
                st.image(processed_image, caption=f"Processed Image {idx + 1}", use_column_width=True)

        else:
            image = demo_image
            st.sidebar.text('Input Image')
            st.sidebar.image(image)
            processed_images = image_stream(image)
            for idx, processed_image in enumerate(processed_images):
                st.image(processed_image, caption=f"Processed Image {idx + 1}", use_column_width=True)


    elif app_mode == "Run on Video":
        use_webcam = st.sidebar.checkbox('Use Webcam')
        video_file = st.sidebar.file_uploader('Upload an Video', type=['mp4', 'avi', 'mov', 'asf'])
        demo_video = 'uploads/video.mp4'
        tffile = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        if not video_file:
            if use_webcam:
                tffile.name = 0
            else:
                tffile.name = demo_video
                demo_vid = open(tffile.name, 'rb')
                demo_bytes = demo_vid.read()
                st.sidebar.text('Input Video')
                st.sidebar.video(demo_bytes)
        else:
            tffile.write(video_file.read())
            demo_vid = open(tffile.name, 'rb')
            demo_bytes = demo_vid.read()
            st.sidebar.text('Input Video')
            st.sidebar.video(demo_bytes)
        video_stream(tffile.name)



if __name__ == "__main__":
    main()