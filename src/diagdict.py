import xml.etree.ElementTree as ET
xml_path = "icd10cm_tabular_2024.xml"

# Parse the XML file
tree = ET.parse(xml_path)
root = tree.getroot()

# Recursive function to extract 'diag' information based on XSD structure
def extract(element):
    diag_dict = {
        'name': element.findtext('name'),
        'desc': element.findtext('desc')
    }

    # Define optional elements based on schema
    opt_elements = [
        'inclusionTerm', 'sevenChrNote', 'includes', 'excludes1', 
        'excludes2', 'codeFirst', 'useAdditionalCode', 'codeAlso', 'notes'
    ]

    # Capture all optional elements if present
    for tag in opt_elements:
        found_element = element.find(tag)
        if found_element is not None:
            # Handle lists of notes in includes, excludes, etc.
            if tag in opt_elements: #['inclusionTerm', 'sevenChrNote', 'includes', 'excludes1', 'excludes2']:
                diag_dict[tag] = [note.text for note in found_element.findall('note')]
            else:
                diag_dict[tag] = found_element.text if found_element.text else None

    sub_diagnoses = element.findall('diag')
    if sub_diagnoses:
        diag_dict['is_leaf'] = False
    else:
        diag_dict['is_leaf'] = True
    #    diag_dict['sub_diagnoses'] = [extract_diagnosis(sub_diag) for sub_diag in sub_diagnoses]

    return diag_dict

def extractSec(element):
    sec_dict = {
        'desc': element.findtext('desc')
    }

    # Define optional elements based on schema
    opt_elements = [
        'inclusionTerm', 'sevenChrNote', 'includes', 'excludes1', 
        'excludes2', 'codeFirst', 'useAdditionalCode', 'codeAlso', 'notes'
    ]

    # Capture all optional elements if present
    for tag in opt_elements:
        found_element = element.find(tag)
        if found_element is not None:
            # Handle lists of notes in includes, excludes, etc.
            if tag in opt_elements: #['inclusionTerm', 'sevenChrNote', 'includes', 'excludes1', 'excludes2']:
                sec_dict[tag] = [note.text for note in found_element.findall('note')]
            else:
                sec_dict[tag] = found_element.text if found_element.text else None
    return sec_dict

# Extract all chapters
dd = {extract(ch)['name']:extract(ch) for ch in root.findall('.//chapter')}

for child in root:
    for gchild in child:
        if child.tag == "chapter":
            if gchild.tag == 'name':
                temp = dd[gchild.text]
                temp['sections'] = {}
            if gchild.tag == "section":
                temp['sections'][gchild.attrib['id']] = extractSec(gchild)
                temp['sections'][gchild.attrib['id']]['categories'] = {} 
                for category in gchild:
                    if category.tag == 'diag':
                        temp['sections'][gchild.attrib['id']]['categories'][category.findtext('name')] = extract(category)
                        for sub4cat in category:
                            if sub4cat.tag == 'diag':
                                temp['sections'][gchild.attrib['id']]['categories'][category.findtext('name')][sub4cat.findtext('name')] =  extract(sub4cat)
                                for sub5cat in sub4cat:
                                    if sub5cat.tag == 'diag':
                                        temp['sections'][gchild.attrib['id']]['categories'][category.findtext('name')][sub4cat.findtext('name')][sub5cat.findtext('name')]  =  extract(sub5cat)
                                        for sub6cat in sub5cat:
                                            if sub6cat.tag == 'diag':
                                                temp['sections'][gchild.attrib['id']]['categories'][category.findtext('name')][sub4cat.findtext('name')][sub5cat.findtext('name')][sub6cat.findtext('name')]  =  extract(sub6cat)
                                                for sub7cat in sub6cat:
                                                    if sub7cat.tag == 'diag':
                                                        temp['sections'][gchild.attrib['id']]['categories'][category.findtext('name')][sub4cat.findtext('name')][sub5cat.findtext('name')][sub6cat.findtext('name')][sub7cat.findtext('name')]  =  extract(sub6cat)
