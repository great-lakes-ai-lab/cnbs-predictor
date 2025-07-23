import json
import os

def generate_notices_file(json_input_file, output_file="Notices.txt"):
    """
    Generates a Notices.txt file from pip-licenses JSON output.
    """
    
    # Add your project's main copyright notice
    notices_content = """cnbs-predictor
Copyright © 2024 The Regents of the University of Michigan
Great Lakes AI Lab

--------------------------------------------------------------------------------
"""

    try:
        with open(json_input_file, 'r', encoding='utf-8') as f:
            dependencies = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON input file '{json_input_file}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_input_file}'. Ensure it's valid JSON.")
        return

    for dep in dependencies:
        name = dep.get("Name", "UNKNOWN_NAME")
        version = dep.get("Version", "UNKNOWN_VERSION")
        license_type = dep.get("License", "UNKNOWN_LICENSE")
        url = dep.get("URL", "UNKNOWN_URL")
        license_text = dep.get("LicenseText", "No license text found.")

        # Clean up common newline issues from JSON output for plain text
        license_text = license_text.replace('\\n', '\n').strip()

        # Try to extract copyright from the beginning of LicenseText if not explicitly listed
        copyright_line = ""
        if "Copyright" not in license_text and "copyright" not in license_text:
            # Simple attempt to find a common copyright pattern
            first_lines = license_text.split('\n')[:5] # Check first few lines
            for line in first_lines:
                if "Copyright" in line or "copyright" in line:
                    copyright_line = line.strip() + "\n"
                    break
        elif "Copyright" in license_text or "copyright" in license_text:
             # If copyright is present, try to extract first instance on its own line
             for line in license_text.split('\n'):
                 if "Copyright" in line or "copyright" in line:
                     copyright_line = line.strip() + "\n"
                     break
        
        notices_content += f"This product includes software from the {name} project (Version: {version}, {license_type})\n"
        if url and url != "UNKNOWN_URL":
            notices_content += f"{url}\n"
        if copyright_line:
             notices_content += f"{copyright_line}\n"
        
        notices_content += f"{license_text}\n\n"
        notices_content += "--------------------------------------------------------------------------------\n"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(notices_content)

    print(f"'{output_file}' generated successfully with {len(dependencies)} dependency entries.")

# --- How to use the script ---
if __name__ == "__main__":
    # Make sure this matches the file name you used when running pip-licenses
    # E.g., if you ran: pip-licenses --format=json --with-license-file > my_licenses.json
    json_file_name = "notices_dependencies.json" # Change this if your file is named differently

    # Create a dummy JSON file for demonstration if it doesn't exist
    # In your actual use, you'd already have this file from pip-licenses
    if not os.path.exists(json_file_name):
        dummy_json_content = """
        [
            {
                "License": "MIT License",
                "LicenseFile": "/path/to/brotli/LICENSE",
                "LicenseText": "Copyright (c) 2009, 2010, 2013-2016 by the Brotli Authors.\\n\\nPermission is hereby granted, free of charge, to any person obtaining a copy\\nof this software and associated documentation files (the \\"Software\\"), to deal\\nin the Software without restriction, including without limitation the rights\\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell\\ncopies of the Software, and to permit persons to whom the Software is\\nfurnished to do so, subject to the following conditions:\\n\\nThe above copyright notice and this permission notice shall be included in\\nall copies or substantial portions of the Software.\\n\\nTHE SOFTWARE IS PROVIDED \\"AS IS\\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\\nIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\\nFITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE\\nAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\\nLIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\\nOUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN\\nTHE SOFTWARE.\\n",
                "Name": "Brotli",
                "URL": "https://github.com/google/brotli",
                "Version": "1.1.0"
            },
            {
                "License": "BSD License",
                "LicenseFile": "/path/to/jinja2/LICENSE.txt",
                "LicenseText": "Copyright 2007 Pallets\\n\\nRedistribution and use in source and binary forms, with or without\\nmodification, are permitted provided that the following conditions are\\nmet:\\n\\n1.  Redistributions of source code must retain the above copyright\\n    notice, this list of conditions and the following disclaimer.\\n\\n2.  Redistributions in binary form must reproduce the above copyright\\n    notice, this list of conditions and the following disclaimer in the\\n    documentation and/or other materials provided with the distribution.\\n\\n3.  Neither the name of the copyright holder nor the names of its\\n    contributors may be used to endorse or promote products derived from\\n    this software without specific prior written permission.\\n\\nTHIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS\\n\\"AS IS\\" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT\\nLIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A\\nPARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT\\nHOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,\\nSPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED\\nTO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR\\nPROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF\\nLIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING\\nNEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS\\nSOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.\\n",
                "Name": "Jinja2",
                "URL": "https://github.com/pallets/jinja/",
                "Version": "3.1.6"
            }
        ]
        """
        with open(json_file_name, 'w', encoding='utf-8') as f:
            f.write(dummy_json_content)
        print(f"Created a dummy '{json_file_name}' for demonstration purposes.")

    generate_notices_file(json_file_name)