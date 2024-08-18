import streamlit as st
import os
from functions import ImageOrganizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the ImageOrganizer
organizer = ImageOrganizer()

def main():
    st.title("Phi3oto Organiser")
    logger.info("Application started")

    # Set up the organizer
    organizer.setup()

    # User input for categories
    st.subheader("Define Categories")
    categories_input = st.text_input("Enter categories separated by commas (e.g., Nature, Urban, People):")
    categories = [cat.strip() for cat in categories_input.split(',') if cat.strip()]
    if categories:
        organizer.set_categories(categories)
        st.success(f"Categories set: {', '.join(categories)}")

    # Load images from the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    organizer.load_directory(current_dir)
    st.info(f"Loaded {len(organizer.files)} images from the current directory")
    
    if st.button("Process and Organize Images"):
        if not organizer.files:
            st.warning("No image files found in the current directory.")
            logger.warning("No image files found in the current directory")
        elif not categories:
            st.warning("Please define categories before processing images.")
            logger.warning("No categories defined")
        else:
            try:
                with st.spinner("Processing images..."):
                    st.text(f"Processing {len(organizer.files)} images")
                    logger.info(f"Starting to process {len(organizer.files)} images")
                    
                    # Prepare files
                    organizer.prepare_files()
                    st.text(f"Files prepared: {len(organizer.prepared_files)} out of {len(organizer.files)}")
                    logger.info(f"File preparation completed. {len(organizer.prepared_files)} files prepared.")
                    
                    if not organizer.prepared_files:
                        st.warning("No files were successfully prepared. Please check the logs for details.")
                        logger.warning("No files were successfully prepared")
                        return

                    # Organize the images
                    organized_images = organizer.organize_images()
                    st.text("Images organized successfully")
                    logger.info("Image organization completed")

                    # Process files (rename, move, add EXIF)
                    organizer.process_files(organized_images, current_dir)
                    st.success("Images processed, renamed, and moved to category folders")

                # Display results
                st.subheader("Organized Images")
                for category, images in organized_images.items():
                    st.write(f"**{category}**")
                    logger.info(f"Displaying category: {category} with {len(images)} images")
                    for old_path, info in images:
                        new_path = os.path.join(current_dir, category, f"{info['name']}{os.path.splitext(old_path)[1]}")
                        st.image(new_path, caption=f"{info['name']} - {info['description']}", use_column_width=True)

            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)
                logger.exception("An unexpected error occurred")
                st.info("Please check your configuration and try again.")

if __name__ == "__main__":
    main()
