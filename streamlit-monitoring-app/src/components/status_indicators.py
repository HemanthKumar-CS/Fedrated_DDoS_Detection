from streamlit import markdown, container

def display_status_indicator(status):
    with container():
        if status == "normal":
            markdown("<h1 style='color: green;'>System Status: Normal</h1>", unsafe_allow_html=True)
        elif status == "alert":
            markdown("<h1 style='color: red;'>System Status: Alert</h1>", unsafe_allow_html=True)
        elif status == "detection":
            markdown("<h1 style='color: orange;'>System Status: Detection in Progress</h1>", unsafe_allow_html=True)
        else:
            markdown("<h1 style='color: gray;'>System Status: Unknown</h1>", unsafe_allow_html=True)

def update_status_indicator(status):
    display_status_indicator(status)