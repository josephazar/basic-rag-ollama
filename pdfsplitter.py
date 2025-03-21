import os
import pikepdf
from pathlib import Path

def split_pdf_by_page(input_folder, output_folder):
    """
    Split PDFs in the input folder by page and save each page as a separate PDF
    in the output folder with the naming pattern filename-page1.pdf, filename-page2.pdf, etc.
    
    Args:
        input_folder (str): Folder containing the PDFs to split
        output_folder (str): Folder where the split PDFs will be saved
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all PDF files in the input folder
    try:
        pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
    except FileNotFoundError:
        print(f"Error: Input folder '{input_folder}' not found.")
        return
    
    if not pdf_files:
        print(f"No PDF files found in '{input_folder}'.")
        return
    
    for pdf_file in pdf_files:
        input_path = os.path.join(input_folder, pdf_file)
        filename = os.path.splitext(pdf_file)[0]  # Get filename without extension
        
        try:
            # Open the PDF file
            with pikepdf.open(input_path) as pdf:
                num_pages = len(pdf.pages)
                print(f"Processing {pdf_file} ({num_pages} pages)...")
                
                # Extract each page and save as a separate PDF
                for page_num in range(num_pages):
                    output_filename = f"{filename}-page{page_num + 1}.pdf"
                    output_path = os.path.join(output_folder, output_filename)
                    
                    try:
                        # Create a new PDF with just this page
                        dst = pikepdf.new()
                        dst.pages.append(pdf.pages[page_num])
                        
                        # Save the page to a new file
                        dst.save(output_path)
                        
                        if (page_num + 1) % 10 == 0 or page_num + 1 == num_pages:
                            print(f"  Processed {page_num + 1}/{num_pages} pages...")
                            
                    except Exception as e:
                        print(f"  Error processing page {page_num + 1} of '{pdf_file}': {str(e)}")
            
            print(f"Completed processing {pdf_file}")
            
        except pikepdf.PdfError as e:
            print(f"Error: Could not read '{pdf_file}'. The file may be corrupted, invalid, or encrypted: {str(e)}")
        except Exception as e:
            print(f"Error processing '{pdf_file}': {str(e)}")

if __name__ == "__main__":
    # Define input and output folders
    input_folder = "docs-raw"
    output_folder = "docs"
    
    # Run the function
    split_pdf_by_page(input_folder, output_folder)
    print("PDF splitting completed!")