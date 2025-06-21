import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from sklearn.datasets import fetch_openml

# Set page configuration
st.set_page_config(
    page_title="MNIST Digit Viewer",
    page_icon="🔢",
    layout="wide"
)

@st.cache_data
def load_mnist_data():
    """Load and cache MNIST dataset using sklearn"""
    try:
        # Load MNIST dataset from sklearn
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        
        # Get images and labels
        x_data = mnist.data.reshape(-1, 28, 28)  # Reshape to 28x28 images
        y_data = mnist.target.astype(int)  # Convert labels to integers
        
        return x_data, y_data
    except Exception as e:
        st.error(f"Error loading MNIST dataset: {e}")
        return None, None

def get_digit_samples(x_data, y_data, digit, num_samples=5):
    """Get random samples of a specific digit"""
    if x_data is None or y_data is None:
        return []
    
    # Find all indices where the label matches the requested digit
    digit_indices = np.where(y_data == digit)[0]
    
    if len(digit_indices) == 0:
        return []
    
    # Randomly select the requested number of samples
    selected_indices = random.sample(list(digit_indices), min(num_samples, len(digit_indices)))
    
    return [x_data[i] for i in selected_indices]

def display_images(images, digit):
    """Display images in a grid"""
    if not images:
        st.error(f"No samples found for digit {digit}")
        return
    
    # Create columns for displaying images
    cols = st.columns(5)
    
    for i, img in enumerate(images):
        with cols[i]:
            # Create figure for matplotlib
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            ax.set_title(f'Sample {i+1}', fontsize=10)
            
            # Display the plot in streamlit
            st.pyplot(fig)
            plt.close(fig)

def main():
    st.title("🔢 MNIST Handwritten Digit Viewer")
    st.markdown("---")
    
    # Add description
    st.markdown("""
    ## About MNIST Dataset
    The MNIST dataset contains 70,000 handwritten digits (60,000 training + 10,000 testing images).
    Each image is 28×28 pixels in grayscale format, representing digits from 0 to 9.
    
    **Instructions:** Enter a digit (0-9) below to see 5 random samples from the MNIST dataset.
    """)
    
    # Load data
    with st.spinner("Loading MNIST dataset..."):
        x_data, y_data = load_mnist_data()
    
    if x_data is None:
        st.error("Failed to load MNIST dataset. Please check your internet connection.")
        return
    
    st.success(f"✅ MNIST dataset loaded successfully! Total samples: {len(x_data)}")
    
    # User input
    st.markdown("---")
    col1, col2 = st.columns([1, 3])
    
    with col1:
        digit_input = st.selectbox(
            "Select a digit to view:",
            options=list(range(10)),
            index=0,
            help="Choose a digit from 0 to 9"
        )
    
    with col2:
        if st.button("🔍 Generate 5 Random Samples", type="primary"):
            st.session_state.generate_samples = True
            st.session_state.selected_digit = digit_input
    
    # Alternative number input
    st.markdown("**Or enter digit manually:**")
    manual_input = st.number_input(
        "Enter digit (0-9):",
        min_value=0,
        max_value=9,
        value=0,
        step=1
    )
    
    if st.button("🎲 Show Samples for Manual Input"):
        st.session_state.generate_samples = True
        st.session_state.selected_digit = manual_input
    
    # Display samples if requested
    if hasattr(st.session_state, 'generate_samples') and st.session_state.generate_samples:
        digit = st.session_state.selected_digit
        
        st.markdown("---")
        st.subheader(f"📊 5 Random Samples of Digit: **{digit}**")
        
        with st.spinner(f"Finding samples for digit {digit}..."):
            samples = get_digit_samples(x_data, y_data, digit, 5)
        
        if samples:
            # Count total occurrences of this digit
            total_count = np.sum(y_data == digit)
            st.info(f"Total samples of digit '{digit}' in dataset: {total_count}")
            
            # Display the images
            display_images(samples, digit)
            
            # Add refresh button
            if st.button("🔄 Get Different Samples"):
                st.rerun()
        else:
            st.error(f"No samples found for digit {digit}")
      # Additional information
    st.markdown("---")
    st.markdown("""
    ### 📚 Dataset Statistics
    - **Total Images:** 70,000 (60,000 training + 10,000 testing)
    - **Image Size:** 28×28 pixels
    - **Color:** Grayscale
    - **Classes:** 10 digits (0-9)
    - **Format:** Normalized pixel values between 0-255
    """)

if __name__ == "__main__":
    main()
