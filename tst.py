import streamlit as st
import os


def load_data(file_path):
    # Use os.path.join to create the file path
    data_path = os.path.join(os.getcwd(), file_path)

    # Check if the file exists
    if os.path.exists(data_path):
        # Load the data
        data = data_path
        return data
    else:
        st.error("File not found: {}".format(data_path))


def main():
    st.title("My Streamlit App")

    # Assume the data file is located in the same directory as the app
    data_file = "Photo/moey.png"

    # Load the data using the relative file path
    data = load_data(data_file)

    # Display the data in the app
    st.write(data)
    st.image(data)


if __name__ == "__main__":
    main()
