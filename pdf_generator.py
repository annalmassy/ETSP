from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import Color, HexColor
from io import BytesIO

def create_header_footer():
    """Create color for header and footer"""
    return HexColor('#1e88e5')

def create_flashcard(front, back, width, height):
    """
    Create a flashcard table with front and back content.
    
    Args:
        front: Front side content
        back: Back side content
        width: Card width
        height: Card height
        
    Returns:
        Table object representing the flashcard
    """
    card_style = ParagraphStyle(
        'CardStyle',
        parent=getSampleStyleSheet()['Normal'],
        fontSize=10,
        leading=12,
        alignment=1
    )
    
    data = [
        [Paragraph(front, card_style)],
        [Paragraph(f"Answer: {back}", card_style)]
    ]
    
    card = Table(data, colWidths=[width], rowHeights=[height/2, height/2])
    card.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (0, 0), HexColor('#e3f2fd')),
        ('BACKGROUND', (0, 1), (0, 1), HexColor('#bbdefb')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    return card

def generate_pdf(content):
    """Generate PDF with study materials."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=20,
        leftMargin=20,
        topMargin=20,
        bottomMargin=20
    )
    
    styles = getSampleStyleSheet()
    story = []
    
    # Define styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        textColor=colors.black,
        spaceAfter=15
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading1'],
        fontSize=14,
        textColor=colors.black,
        spaceBefore=8,
        spaceAfter=4
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        leading=12,
        spaceBefore=4,
        spaceAfter=4
    )
    
    # Add title
    story.append(Paragraph(content['title'], title_style))
    story.append(Spacer(1, 10))
    
    # Create two-column layout for study guide
    study_guide_content = []
    sections = [
        ("Overview and Introduction", content['study_guide']['overview']),
        ("Core Concepts and Fundamentals", content['study_guide']['core_concepts']),
        ("Technical Details and Methodology", content['study_guide']['technical_details']),
        ("Practical Applications", content['study_guide']['practical_applications']),
        ("Challenges and Limitations", content['study_guide']['challenges']),
        ("Future Directions and Trends", content['study_guide']['future_directions'])
    ]
    
    for section_title, section_content in sections:
        study_guide_content.append(Paragraph(section_title, heading_style))
        study_guide_content.append(Paragraph(section_content, body_style))
        study_guide_content.append(Spacer(1, 8))
    
    # Split content into two columns
    col_width = (doc.width - 40) / 2
    study_guide_table = Table(
        [[study_guide_content[:len(study_guide_content)//2],
          study_guide_content[len(study_guide_content)//2:]]],
        colWidths=[col_width, col_width]
    )
    study_guide_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(study_guide_table)
    
    # Add flashcards in 3x4 grid
    story.append(PageBreak())
    story.append(Paragraph("Flashcards", title_style))
    
    # Calculate card dimensions for 3x4 grid
    card_width = (doc.width - 60) / 3
    card_height = (doc.height - 80) / 4
    
    # Create flashcard grid
    for i in range(0, 12, 3):
        row_cards = content['flashcards'][i:i+3]
        card_row = []
        for card in row_cards:
            flashcard = create_flashcard(card['front'], card['back'], card_width, card_height)
            card_row.append(flashcard)
        
        row_table = Table([card_row], 
                         colWidths=[card_width] * 3,
                         rowHeights=[card_height])
        row_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(row_table)
        story.append(Spacer(1, 8))
    
    # Add exercises
    story.append(PageBreak())
    story.append(Paragraph("Exercises", title_style))
    for i, exercise in enumerate(content['exercises'], 1):
        story.append(Paragraph(f"Exercise {i}:", heading_style))
        story.append(Paragraph(exercise['question'], body_style))
        story.append(Spacer(1, 8))
        story.append(Paragraph("Solution:", heading_style))
        story.append(Paragraph(exercise['solution'], body_style))
        story.append(Spacer(1, 12))
    
    doc.build(story)
    return buffer.getvalue()