from html import escape
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from PIL import Image, ImageDraw, ImageFont


OUT = Path("Ecommerce_Recommender_System_Final_Documentation.docx")
ASSET_DIR = Path("documentation_assets")


def font(size=26, bold=False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def wrap_text(draw, text, fnt, width):
    words = str(text).split()
    lines = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textbbox((0, 0), trial, font=fnt)[2] <= width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def rounded_box(draw, xy, fill, outline="#1f2937", radius=18, width=3):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def centered_text(draw, box, text, fnt, fill="#111827"):
    x1, y1, x2, y2 = box
    lines = wrap_text(draw, text, fnt, x2 - x1 - 28)
    line_h = fnt.size + 8
    y = y1 + ((y2 - y1) - line_h * len(lines)) / 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=fnt)
        draw.text((x1 + ((x2 - x1) - (bbox[2] - bbox[0])) / 2, y), line, font=fnt, fill=fill)
        y += line_h


def arrow(draw, start, end, fill="#334155", width=4):
    draw.line([start, end], fill=fill, width=width)
    ex, ey = end
    sx, sy = start
    if abs(ex - sx) >= abs(ey - sy):
        direction = 1 if ex > sx else -1
        pts = [(ex, ey), (ex - 16 * direction, ey - 9), (ex - 16 * direction, ey + 9)]
    else:
        direction = 1 if ey > sy else -1
        pts = [(ex, ey), (ex - 9, ey - 16 * direction), (ex + 9, ey - 16 * direction)]
    draw.polygon(pts, fill=fill)


def save_canvas(name, title, subtitle=None):
    ASSET_DIR.mkdir(exist_ok=True)
    img = Image.new("RGB", (1400, 820), "#f8fafc")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 1400, 92), fill="#0f172a")
    draw.text((42, 24), title, font=font(34, True), fill="#ffffff")
    if subtitle:
        draw.text((42, 62), subtitle, font=font(18), fill="#cbd5e1")
    return img, draw


def make_architecture():
    img, draw = save_canvas(
        "architecture.png",
        "System Architecture",
        "React client, Flask services, recommendation engine, SQLite, and Amazon catalog data",
    )
    boxes = {
        "React Frontend": (70, 190, 360, 320),
        "Flask API": (555, 190, 845, 320),
        "SQLite Database": (1035, 190, 1325, 320),
        "Product Service": (260, 500, 550, 630),
        "Hybrid Recommender": (555, 500, 845, 630),
        "Amazon CSV Dataset": (850, 500, 1140, 630),
    }
    fills = ["#dbeafe", "#dcfce7", "#fee2e2", "#fef3c7", "#e0e7ff", "#fce7f3"]
    for (label, xy), fill_color in zip(boxes.items(), fills):
        rounded_box(draw, xy, fill_color)
        centered_text(draw, xy, label, font(26, True))
    arrow(draw, (360, 255), (555, 255))
    arrow(draw, (845, 255), (1035, 255))
    arrow(draw, (700, 320), (700, 500))
    arrow(draw, (850, 565), (845, 565))
    arrow(draw, (550, 565), (555, 565))
    arrow(draw, (405, 500), (230, 320))
    path = ASSET_DIR / "architecture.png"
    img.save(path)
    return path


def make_workflow():
    img, draw = save_canvas(
        "workflow.png",
        "Recommendation Workflow",
        "How a user request becomes a ranked list of products",
    )
    steps = [
        "User selects a product",
        "Clean catalog text",
        "Build TF-IDF vectors",
        "Calculate content similarity",
        "Read user rating history",
        "Calculate CF scores",
        "Blend CB, CF, profile, popularity",
        "Return ranked recommendations",
    ]
    x, y = 58, 170
    for i, step in enumerate(steps):
        bx = x + (i % 4) * 330
        by = y + (i // 4) * 260
        box = (bx, by, bx + 270, by + 115)
        rounded_box(draw, box, "#ecfeff" if i < 4 else "#fff7ed")
        centered_text(draw, box, step, font(23, True))
        if i % 4 != 3:
            arrow(draw, (bx + 270, by + 58), (bx + 330, by + 58))
    arrow(draw, (1288, 228), (1288, 430))
    arrow(draw, (1288, 430), (1138, 430))
    for i in range(4, 7):
        bx = x + (i % 4) * 330
        by = y + (i // 4) * 260
        arrow(draw, (bx + 270, by + 58), (bx + 330, by + 58))
    path = ASSET_DIR / "workflow.png"
    img.save(path)
    return path


def make_use_case():
    img, draw = save_canvas("use_case.png", "Use Case Diagram", "Main actions supported by the application")
    draw.ellipse((80, 210, 160, 290), fill="#bfdbfe", outline="#1e3a8a", width=3)
    draw.line((120, 290, 120, 440), fill="#1e3a8a", width=5)
    draw.line((60, 350, 180, 350), fill="#1e3a8a", width=5)
    draw.line((120, 440, 70, 560), fill="#1e3a8a", width=5)
    draw.line((120, 440, 170, 560), fill="#1e3a8a", width=5)
    draw.text((68, 590), "Customer", font=font(24, True), fill="#111827")
    actions = [
        "Register / Login",
        "Search Products",
        "View Product Details",
        "Get Recommendations",
        "Manage Wishlist",
        "Manage Cart",
    ]
    for i, action_label in enumerate(actions):
        cx = 480 + (i % 2) * 420
        cy = 200 + (i // 2) * 170
        draw.ellipse((cx, cy, cx + 300, cy + 90), fill="#dcfce7", outline="#166534", width=3)
        centered_text(draw, (cx, cy, cx + 300, cy + 90), action_label, font(22, True))
        arrow(draw, (190, 365), (cx, cy + 45), width=2)
    draw.ellipse((1180, 250, 1260, 330), fill="#fecaca", outline="#991b1b", width=3)
    draw.line((1220, 330, 1220, 480), fill="#991b1b", width=5)
    draw.line((1160, 390, 1280, 390), fill="#991b1b", width=5)
    draw.line((1220, 480, 1170, 600), fill="#991b1b", width=5)
    draw.line((1220, 480, 1270, 600), fill="#991b1b", width=5)
    draw.text((1185, 630), "Admin", font=font(24, True), fill="#111827")
    admin_box = (870, 620, 1110, 710)
    draw.ellipse(admin_box, fill="#fee2e2", outline="#991b1b", width=3)
    centered_text(draw, admin_box, "View Users", font(22, True))
    arrow(draw, (1180, 475), (1110, 665), width=2)
    path = ASSET_DIR / "use_case.png"
    img.save(path)
    return path


def make_data_model():
    img, draw = save_canvas("data_model.png", "Database and Data Model", "Application tables and dataset fields")
    tables = [
        ("users", ["id", "username", "email", "password_hash", "role", "created_at"], (70, 160, 400, 430), "#dbeafe"),
        ("token_sessions", ["id", "jti", "user_id", "issued_at", "expires_at", "revoked_at"], (525, 160, 855, 430), "#dcfce7"),
        ("wishlist_items", ["id", "user_id", "product_id", "created_at"], (980, 160, 1310, 390), "#fef3c7"),
        ("cart_items", ["id", "user_id", "product_id", "quantity", "updated_at"], (290, 520, 620, 745), "#fee2e2"),
        ("amazon.csv", ["product_id", "product_name", "category", "rating", "about_product", "img_link"], (780, 500, 1180, 755), "#e0e7ff"),
    ]
    for name, cols, xy, fill_color in tables:
        rounded_box(draw, xy, fill_color)
        draw.rectangle((xy[0], xy[1], xy[2], xy[1] + 48), fill="#111827")
        draw.text((xy[0] + 18, xy[1] + 10), name, font=font(23, True), fill="#ffffff")
        y = xy[1] + 70
        for col in cols:
            draw.text((xy[0] + 22, y), col, font=font(20), fill="#111827")
            y += 31
    arrow(draw, (400, 280), (525, 280))
    arrow(draw, (400, 320), (980, 275))
    arrow(draw, (400, 360), (455, 520))
    arrow(draw, (620, 632), (780, 632))
    arrow(draw, (980, 315), (910, 500))
    path = ASSET_DIR / "data_model.png"
    img.save(path)
    return path


def make_sequence():
    img, draw = save_canvas("sequence.png", "Sequence Diagram", "Personalized recommendation request")
    actors = ["Customer", "React App", "Flask API", "Auth Middleware", "Recommender", "Dataset"]
    xs = [120, 360, 600, 840, 1080, 1280]
    for actor, x in zip(actors, xs):
        rounded_box(draw, (x - 82, 145, x + 82, 200), "#e2e8f0")
        centered_text(draw, (x - 82, 145, x + 82, 200), actor, font(18, True))
        draw.line((x, 200, x, 735), fill="#94a3b8", width=2)
    messages = [
        (0, 1, 250, "open product"),
        (1, 2, 310, "GET recommendations"),
        (2, 3, 370, "validate JWT"),
        (2, 4, 450, "request hybrid scores"),
        (4, 5, 510, "load product and ratings"),
        (4, 2, 590, "ranked products"),
        (2, 1, 650, "JSON response"),
        (1, 0, 710, "show products"),
    ]
    for src, dst, y, label in messages:
        arrow(draw, (xs[src] + (22 if dst > src else -22), y), (xs[dst] - (22 if dst > src else -22), y), width=3)
        draw.text((min(xs[src], xs[dst]) + 30, y - 28), label, font=font(17), fill="#111827")
    path = ASSET_DIR / "sequence.png"
    img.save(path)
    return path


def make_deployment():
    img, draw = save_canvas("deployment.png", "Deployment Diagram", "Local development and production-ready layout")
    boxes = [
        ("Browser", "React/Vite user interface", (90, 260, 390, 430), "#dbeafe"),
        ("Web Server", "Static React build", (520, 160, 850, 330), "#dcfce7"),
        ("Flask Server", "API, auth, recommender", (520, 485, 850, 655), "#fef3c7"),
        ("Database", "SQLite now, external DB later", (1010, 330, 1310, 500), "#fee2e2"),
    ]
    for title, sub, xy, fill_color in boxes:
        rounded_box(draw, xy, fill_color)
        centered_text(draw, (xy[0], xy[1] + 20, xy[2], xy[1] + 80), title, font(26, True))
        centered_text(draw, (xy[0] + 12, xy[1] + 85, xy[2] - 12, xy[3] - 20), sub, font(21))
    arrow(draw, (390, 345), (520, 245))
    arrow(draw, (390, 345), (520, 570))
    arrow(draw, (850, 570), (1010, 415))
    arrow(draw, (850, 245), (1010, 415))
    path = ASSET_DIR / "deployment.png"
    img.save(path)
    return path


def make_results_chart():
    img, draw = save_canvas("results_chart.png", "Evaluation Results", "Offline ranking metrics for CF, CB, and Hybrid")
    metrics = ["P@5", "R@5", "F1@5", "NDCG@5", "P@10", "R@10", "F1@10", "NDCG@10"]
    cb = [0.1481, 0.5185, 0.2305, 0.6323, 0.0727, 0.5909, 0.1295, 0.3894]
    hybrid = [0.1462, 0.5192, 0.2281, 0.4412, 0.0864, 0.5227, 0.1482, 0.3433]
    left, top, bottom = 110, 160, 700
    draw.line((left, top, left, bottom), fill="#111827", width=3)
    draw.line((left, bottom, 1320, bottom), fill="#111827", width=3)
    scale = 470
    group_w = 145
    for i, metric in enumerate(metrics):
        x = left + 40 + i * group_w
        cb_h = cb[i] * scale
        hy_h = hybrid[i] * scale
        draw.rectangle((x, bottom - cb_h, x + 38, bottom), fill="#2563eb")
        draw.rectangle((x + 45, bottom - hy_h, x + 83, bottom), fill="#16a34a")
        draw.text((x - 4, bottom + 18), metric, font=font(18, True), fill="#111827")
    draw.rectangle((1045, 145, 1085, 170), fill="#2563eb")
    draw.text((1098, 142), "Content-Based", font=font(20), fill="#111827")
    draw.rectangle((1045, 185, 1085, 210), fill="#16a34a")
    draw.text((1098, 182), "Hybrid", font=font(20), fill="#111827")
    for tick in [0.0, 0.2, 0.4, 0.6]:
        y = bottom - tick * scale
        draw.line((left - 8, y, left, y), fill="#111827", width=2)
        draw.text((55, y - 12), f"{tick:.1f}", font=font(17), fill="#111827")
    path = ASSET_DIR / "results_chart.png"
    img.save(path)
    return path


def make_ui_mock():
    img, draw = save_canvas("ui_mock.png", "Application Screens", "Catalog, product details, wishlist, cart, and admin pages")
    draw.rectangle((70, 145, 1330, 220), fill="#111827")
    draw.text((95, 168), "Recommender Store", font=font(26, True), fill="#ffffff")
    draw.text((820, 172), "Products   Wishlist   Cart   Admin", font=font(22), fill="#e5e7eb")
    draw.rounded_rectangle((95, 260, 590, 315), radius=12, fill="#ffffff", outline="#cbd5e1", width=2)
    draw.text((120, 276), "Search products", font=font(22), fill="#64748b")
    draw.rounded_rectangle((650, 260, 1030, 315), radius=12, fill="#ffffff", outline="#cbd5e1", width=2)
    draw.text((675, 276), "All categories", font=font(22), fill="#64748b")
    colors = ["#bfdbfe", "#bbf7d0", "#fecaca", "#fde68a", "#ddd6fe", "#fbcfe8"]
    for i in range(6):
        x = 95 + (i % 3) * 410
        y = 365 + (i // 3) * 175
        draw.rounded_rectangle((x, y, x + 340, y + 135), radius=16, fill="#ffffff", outline="#cbd5e1", width=2)
        draw.rectangle((x + 18, y + 18, x + 118, y + 118), fill=colors[i])
        draw.text((x + 140, y + 25), "Product title", font=font(22, True), fill="#111827")
        draw.text((x + 140, y + 63), "Category / subcategory", font=font(18), fill="#64748b")
        draw.text((x + 140, y + 96), "Price and rating", font=font(18, True), fill="#111827")
    path = ASSET_DIR / "ui_mock.png"
    img.save(path)
    return path


def make_figures():
    return [
        ("Figure 4.2.1 System Architecture", make_architecture()),
        ("Figure 4.2.2 Recommendation Workflow", make_workflow()),
        ("Figure 4.2.3 Use Case Diagram", make_use_case()),
        ("Figure 4.2.4 Data Model Diagram", make_data_model()),
        ("Figure 4.2.5 Sequence Diagram", make_sequence()),
        ("Figure 4.2.6 Deployment Diagram", make_deployment()),
        ("Figure 5.5.1 Application Screens", make_ui_mock()),
        ("Figure 5.6.1 Evaluation Results", make_results_chart()),
    ]


def r(text, bold=False, italic=False):
    props = []
    if bold:
        props.append("<w:b/>")
    if italic:
        props.append("<w:i/>")
    rpr = f"<w:rPr>{''.join(props)}</w:rPr>" if props else ""
    return f"<w:r>{rpr}<w:t xml:space=\"preserve\">{escape(str(text))}</w:t></w:r>"


def p(text="", style=None, align=None, bold=False):
    props = []
    if style:
        props.append(f'<w:pStyle w:val="{style}"/>')
    if align:
        props.append(f'<w:jc w:val="{align}"/>')
    ppr = f"<w:pPr>{''.join(props)}</w:pPr>" if props else ""
    runs = []
    for i, line in enumerate(str(text).split("\n")):
        if i:
            runs.append("<w:r><w:br/></w:r>")
        runs.append(r(line, bold=bold))
    return f"<w:p>{ppr}{''.join(runs)}</w:p>"


def heading(text, level=1):
    return p(text, style=f"Heading{level}")


def bullet(text):
    return f'<w:p><w:pPr><w:pStyle w:val="ListParagraph"/></w:pPr>{r("- " + text)}</w:p>'


def table(headers, rows):
    cols = len(headers)
    col_width = max(1800, int(9000 / cols))
    grid = "".join(f'<w:gridCol w:w="{col_width}"/>' for _ in range(cols))

    def cell(text, bold=False):
        return (
            f'<w:tc><w:tcPr><w:tcW w:w="{col_width}" w:type="dxa"/></w:tcPr>'
            f'<w:p>{r(str(text), bold=bold)}</w:p></w:tc>'
        )

    header_xml = "<w:tr>" + "".join(cell(h, True) for h in headers) + "</w:tr>"
    row_xml = "".join("<w:tr>" + "".join(cell(value) for value in row) + "</w:tr>" for row in rows)
    return (
        '<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/>'
        '<w:tblW w:w="0" w:type="auto"/>'
        '<w:tblBorders><w:top w:val="single" w:sz="4"/><w:left w:val="single" w:sz="4"/>'
        '<w:bottom w:val="single" w:sz="4"/><w:right w:val="single" w:sz="4"/>'
        '<w:insideH w:val="single" w:sz="4"/><w:insideV w:val="single" w:sz="4"/></w:tblBorders>'
        f"</w:tblPr><w:tblGrid>{grid}</w:tblGrid>{header_xml}{row_xml}</w:tbl>"
    )


def page_break():
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def image_block(rel_id, caption, width_inches=6.5, height_inches=3.8):
    cx = int(width_inches * 914400)
    cy = int(height_inches * 914400)
    doc_id = "".join(ch for ch in rel_id if ch.isdigit()) or "1"
    drawing = f'''
<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:drawing>
<wp:inline distT="0" distB="0" distL="0" distR="0">
<wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>
<wp:docPr id="{doc_id}" name="{caption}"/><wp:cNvGraphicFramePr/>
<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
<pic:nvPicPr><pic:cNvPr id="0" name="{caption}.png"/><pic:cNvPicPr/></pic:nvPicPr>
<pic:blipFill><a:blip r:embed="{rel_id}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>
</pic:pic></a:graphicData></a:graphic>
</wp:inline></w:drawing></w:r></w:p>'''
    return drawing + p(caption, align="center")


def chapter_title(number, title):
    return heading(f"CHAPTER {number}: {title}", 1)


def build_body(image_rels):
    body = []
    body.extend([
        p("E-COMMERCE RECOMMENDER SYSTEM", style="Title", align="center"),
        p("Project Report submitted in partial fulfilment of the requirements for the award of the degree", align="center"),
        p("BACHELOR / MASTER OF TECHNOLOGY", align="center", bold=True),
        p("IN", align="center"),
        p("COMPUTER SCIENCE AND ENGINEERING / INFORMATION TECHNOLOGY", align="center", bold=True),
        p("Submitted by", align="center"),
        p("Your Name", align="center", bold=True),
        p("Roll No: Your Roll Number", align="center"),
        p("Under the guidance of", align="center"),
        p("Your Guide Name", align="center", bold=True),
        p("DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING", align="center", bold=True),
        p("Your College / University Name", align="center", bold=True),
        p("2026", align="center"),
        page_break(),
        heading("CERTIFICATE", 1),
        p("This is to certify that the project report entitled E-COMMERCE RECOMMENDER SYSTEM is a bonafide record of project work carried out by Your Name, bearing Roll No: Your Roll Number, in partial fulfilment of the requirements for the award of the degree. The work has been completed under the guidance of the project guide and has not been submitted elsewhere for the award of any other degree."),
        p("\n\nSignature of Guide                                      Signature of HOD"),
        page_break(),
        heading("DECLARATION", 1),
        p("I hereby declare that this project report entitled E-COMMERCE RECOMMENDER SYSTEM is the result of my own project work. The system, documentation, analysis, and implementation presented in this report are based on my understanding of recommendation systems, web application development, and the project source code. I also declare that this report has not been submitted to any other institution for the award of any degree or diploma."),
        p("\nYour Name\nRoll No: Your Roll Number"),
        page_break(),
        heading("ACKNOWLEDGEMENT", 1),
        p("I express my sincere gratitude to my project guide for valuable guidance, encouragement, and support throughout the development of this project. I am also thankful to the Head of the Department, faculty members, and lab staff for providing the required facilities and suggestions. I extend my thanks to my friends and family for their continuous motivation during the completion of this work."),
        p("This project helped me understand how recommendation systems are designed in real applications, how backend APIs communicate with frontend applications, and how user authentication, product catalogs, and recommendation logic can be combined into one working system."),
        p("\nYour Name"),
        page_break(),
        heading("ABSTRACT", 1),
        p("Online shopping platforms contain a large number of products, and users often need help finding items that match their interests. The E-Commerce Recommender System solves this problem by suggesting relevant products using a hybrid recommendation approach. The project combines content-based filtering, collaborative filtering, user profile signals, and popularity-based fallback ranking. Product details such as name, category, description, brand, rating, price, and image link are taken from an Amazon product dataset. Content-based filtering uses TF-IDF and cosine similarity to find products with similar textual features. Collaborative filtering uses user-product rating interactions to identify products that may be useful for a particular user. The hybrid method combines both approaches so that the system can still provide meaningful recommendations even when user history is sparse."),
        p("The application is implemented with a Flask backend, SQLite database, REST API endpoints, JWT authentication, and a React frontend. Users can register, login, browse products, search by category, view product details, receive similar and personalized recommendations, add products to wishlist, and manage cart items. Admin users can view registered users. The model is evaluated using ranking metrics such as Precision@K, Recall@K, F1@K, and NDCG@K. The result shows that the content-based model performs strongly on sparse data, while the hybrid model provides a practical balance for a real e-commerce environment."),
        page_break(),
        heading("TABLE OF CONTENTS", 1),
        table(["SNO", "TOPIC"], [
            ["1", "INTRODUCTION"],
            ["2", "LITERATURE SURVEY"],
            ["3", "ANALYSIS"],
            ["4", "DESIGN"],
            ["5", "IMPLEMENTATION AND RESULTS"],
            ["6", "TESTING AND VALIDATION"],
            ["7", "CONCLUSION"],
            ["8", "FUTURE ENHANCEMENT"],
            ["9", "REFERENCES"],
        ]),
        page_break(),
        heading("LIST OF FIGURES", 1),
        table(["SNO", "FIGURE"], [[str(i + 1), caption] for i, (caption, _) in enumerate(image_rels)]),
        page_break(),
        chapter_title(1, "INTRODUCTION"),
        heading("1.1 Problem Definition", 2),
        p("In an e-commerce website, users are presented with many products across different categories. Searching manually through all products takes time and may not always lead to the best choice. A recommender system is needed to understand product similarity and user preferences so that useful products can be suggested automatically."),
        heading("1.2 Motivation of the Project", 2),
        p("Recommendation systems are used by modern platforms such as online stores, streaming services, and learning portals. They improve user experience by reducing search effort and showing relevant options. This project is motivated by the need to build a practical recommendation system that can work with real product data and can be connected to a complete web application."),
        heading("1.3 Project Objectives", 2),
        bullet("Build an e-commerce web application with product browsing, search, authentication, wishlist, and cart features."),
        bullet("Implement content-based filtering using product name, category, brand, and description."),
        bullet("Implement collaborative filtering using user-product rating interactions."),
        bullet("Combine both methods into a hybrid recommender that handles sparse data."),
        bullet("Evaluate the recommender using standard ranking metrics."),
        page_break(),
        chapter_title(2, "LITERATURE SURVEY"),
        heading("2.1 Introduction", 2),
        p("A recommender system is a software technique that predicts which items may be useful or interesting for a user. In e-commerce, recommendation systems are important because they personalize product discovery. The major approaches are content-based filtering, collaborative filtering, and hybrid filtering."),
        heading("2.1.1 Machine Learning", 2),
        p("Machine learning allows a system to learn patterns from data instead of depending only on fixed rules. In this project, product text and rating interactions are used to compute similarity and ranking scores. The recommender does not simply show random products; it uses measurable signals from the dataset."),
        heading("2.1.2 Recommendation Techniques", 2),
        p("Content-based filtering recommends products similar to a selected product by comparing product features. Collaborative filtering recommends products by learning from user-item interactions. Hybrid filtering combines both methods to reduce limitations such as cold start, sparse ratings, and overspecialized suggestions."),
        heading("2.2 Dataset", 2),
        p("The project uses an Amazon product dataset stored in data/raw/amazon.csv. The dataset contains product_id, product_name, category, about_product, rating, discounted_price, img_link, and user_id. The recommender cleans the data, converts ratings into numeric form, splits multi-value user identifiers, removes duplicate user-product pairs, and prepares a product catalog and interaction table."),
        heading("2.3 Existing System", 2),
        p("In a simple product listing system, users must search manually or browse categories without personalization. Such systems may show popular or recently added products, but they do not understand product similarity or individual user interest. Collaborative filtering alone can also fail when user history is limited."),
        heading("2.4 Proposed System", 2),
        p("The proposed system uses a hybrid recommendation model. It compares products using TF-IDF content vectors and cosine similarity. It also uses user-item rating interactions to compute collaborative scores. These scores are blended with profile-based content scores and popularity scores. The web application exposes this functionality through REST APIs and a React user interface."),
        page_break(),
        chapter_title(3, "ANALYSIS"),
        heading("3.1 System Analysis", 2),
        p("The system has two major parts: the web application and the recommendation engine. The web application manages users, products, wishlist, cart, and admin functions. The recommendation engine prepares product and interaction data, computes similarity matrices, and returns ranked products."),
        heading("3.2 Software Requirements Specification", 2),
        table(["Module", "Requirement"], [
            ["Authentication", "User registration, login, logout, password hashing, JWT validation, and token revocation."],
            ["Product Catalog", "List products, filter by search and category, view full product details."],
            ["Recommendation", "Show similar products and personalized recommendations for logged-in users."],
            ["Wishlist", "Add, list, and remove saved products."],
            ["Cart", "Add products, update quantity, list cart items, and remove products."],
            ["Admin", "View registered users with role-based authorization."],
        ]),
        heading("3.3 Software Requirements", 2),
        table(["Technology", "Purpose"], [
            ["Python", "Backend language and data processing."],
            ["Flask", "Web framework and API server."],
            ["SQLite", "Local application database."],
            ["Pandas and NumPy", "Dataset cleaning and matrix operations."],
            ["scikit-learn", "TF-IDF vectorization and cosine similarity."],
            ["React and Vite", "Frontend application."],
            ["PyJWT", "Token-based authentication."],
        ]),
        heading("3.4 Hardware Requirements", 2),
        bullet("Processor: Intel i3 or above."),
        bullet("RAM: 4 GB minimum, 8 GB recommended."),
        bullet("Storage: At least 1 GB free space for source code, dataset, and database."),
        bullet("Operating System: Windows, Linux, or macOS with Python and Node.js support."),
        heading("3.5 Feasibility Study", 2),
        p("The project is technically feasible because it uses open-source tools and a manageable dataset. It is economically feasible because no paid service is required for local execution. It is operationally feasible because users can interact with the system through familiar e-commerce screens such as catalog, product detail, wishlist, and cart."),
        page_break(),
        chapter_title(4, "DESIGN"),
        heading("4.1 Introduction", 2),
        p("Design explains how the modules of the system are organized and how data moves between them. The application follows a client-server design. The React frontend sends requests to Flask API endpoints. The Flask backend authenticates users, accesses the SQLite database, reads product data, and calls the recommendation engine when needed."),
        image_block("rIdImg1", "Figure 4.2.1 System Architecture"),
        image_block("rIdImg2", "Figure 4.2.2 Recommendation Workflow"),
        image_block("rIdImg3", "Figure 4.2.3 Use Case Diagram"),
        image_block("rIdImg4", "Figure 4.2.4 Data Model Diagram"),
        image_block("rIdImg5", "Figure 4.2.5 Sequence Diagram"),
        image_block("rIdImg6", "Figure 4.2.6 Deployment Diagram"),
        heading("4.3 Module Design", 2),
        table(["File / Folder", "Description"], [
            ["app/api.py", "REST endpoints for authentication, products, recommendations, wishlist, cart, and admin users."],
            ["app/models.py", "SQLAlchemy models for users, token sessions, wishlist items, and cart items."],
            ["app/recommender.py", "Content-based, collaborative, and hybrid recommendation logic."],
            ["app/product_service.py", "Product catalog formatting and helper functions."],
            ["app/evaluate.py", "Evaluation logic for Precision, Recall, F1, NDCG, RMSE, and MAE."],
            ["frontend/src/main.jsx", "React routes and user interface components."],
        ]),
        page_break(),
        chapter_title(5, "IMPLEMENTATION AND RESULTS"),
        heading("5.1 Implementation", 2),
        p("The backend is built using Flask. It provides REST endpoints under /api. Public users can view products and categories, while logged-in users can access similar products, personalized recommendations, wishlist, and cart. JWT tokens are generated during login and registration. Logout is handled by storing token session records and marking the current token as revoked."),
        p("The frontend is built using React. It uses hash-based routing and displays pages for login, registration, catalog, product details, wishlist, cart, and admin users. Product cards show image, title, category, price, and rating. On the product detail page, authenticated users can add products to cart or wishlist and see recommendation rails."),
        heading("5.2 Recommendation Implementation", 2),
        p("The content-based module creates a combined text field from product name, category, brand, and product description. TF-IDF converts this text into numeric vectors, and cosine similarity measures how close two products are. The collaborative module creates a user-item rating matrix and computes both user-user and item-item similarity. The hybrid module normalizes scores and combines content similarity, collaborative score, user profile score, and popularity score."),
        heading("5.3 Sample Code Explanation", 2),
        p("The function hybrid_recommend() receives a user_id and product_id. It first calculates hybrid scores for a wide candidate set. Then it includes strong content-based products, profile-based products, and final hybrid-ranked products. The result is converted into a readable list containing product_id, product_name, rating, discounted_price, and image link."),
        heading("5.4 Application Design", 2),
        image_block("rIdImg7", "Figure 5.5.1 Application Screens"),
        heading("5.5 Results", 2),
        p("The evaluation compares collaborative filtering, content-based filtering, and hybrid filtering. In the recorded evaluation, collaborative filtering alone performs weakly because many users have limited overlapping rating history. Content-based filtering performs strongly because product metadata is rich. The hybrid model gives a balanced result and improves some measures by combining multiple signals."),
        image_block("rIdImg8", "Figure 5.6.1 Evaluation Results"),
        table(["Metric", "CF", "CB", "Hybrid"], [
            ["Precision@5", "0.0000", "0.1481", "0.1462"],
            ["Recall@5", "0.0000", "0.5185", "0.5192"],
            ["F1@5", "0.0000", "0.2305", "0.2281"],
            ["NDCG@5", "0.0000", "0.6323", "0.4412"],
            ["Precision@10", "0.0000", "0.0727", "0.0864"],
            ["Recall@10", "0.0000", "0.5909", "0.5227"],
            ["F1@10", "0.0000", "0.1295", "0.1482"],
            ["NDCG@10", "0.0000", "0.3894", "0.3433"],
        ]),
        page_break(),
        chapter_title(6, "TESTING AND VALIDATION"),
        heading("6.1 Introduction", 2),
        p("Testing checks whether each module works as expected. The project includes API validation, authentication checks, recommendation checks, and basic user workflow testing."),
        heading("6.2 Test Cases and Scenarios", 2),
        table(["Test Case", "Expected Result"], [
            ["Register with valid details", "A new user account is created and a JWT token is returned."],
            ["Login with correct credentials", "The user receives a valid JWT token."],
            ["Access protected API without token", "The API returns a 401 error."],
            ["Logout user", "The active token is revoked and cannot be reused."],
            ["List products", "The API returns product items with title, image, category, price, and rating."],
            ["Request similar products", "The system returns products similar to the selected product."],
            ["Add product to wishlist", "The selected product is saved for the logged-in user."],
            ["Add product to cart", "The selected product appears in the cart with quantity."],
            ["Admin users endpoint as normal user", "The API returns a 403 forbidden error."],
        ]),
        heading("6.3 Validation", 2),
        p("The system is validated by checking API responses, user authentication behavior, recommendation output, and evaluation metrics. The project also uses Python scripts for recommender testing and a React frontend for manual workflow validation."),
        page_break(),
        chapter_title(7, "CONCLUSION"),
        p("The E-Commerce Recommender System successfully demonstrates how recommendation techniques can be integrated into a real web application. It supports user authentication, product browsing, wishlist, cart, admin access, and personalized recommendations. The hybrid model combines content-based filtering and collaborative filtering, making the system more reliable when user interaction data is sparse. The project gives a practical understanding of machine learning, backend development, API design, database modeling, and frontend integration."),
        page_break(),
        chapter_title(8, "FUTURE ENHANCEMENT"),
        bullet("Improve collaborative filtering using matrix factorization or deep learning models."),
        bullet("Add semantic embeddings to understand product meaning better than TF-IDF alone."),
        bullet("Track real-time user actions such as clicks, cart additions, and purchases."),
        bullet("Add payment, order history, and product review modules."),
        bullet("Deploy the system on a cloud platform with a production database."),
        bullet("Add recommendation diversity, novelty, and explainability features."),
        page_break(),
        chapter_title(9, "REFERENCES"),
        p("[1] F. Ricci, L. Rokach, and B. Shapira, Recommender Systems Handbook. Springer, 2015."),
        p("[2] J. B. Schafer, J. Konstan, and J. Riedl, E-Commerce Recommendation Applications, Data Mining and Knowledge Discovery, 2001."),
        p("[3] G. Adomavicius and A. Tuzhilin, Toward the Next Generation of Recommender Systems, IEEE Transactions on Knowledge and Data Engineering, 2005."),
        p("[4] C. D. Manning, P. Raghavan, and H. Schutze, Introduction to Information Retrieval. Cambridge University Press, 2008."),
        p("[5] scikit-learn Developers, scikit-learn: Machine Learning in Python, official documentation."),
        p("[6] Flask Documentation, Pallets Projects."),
        p("[7] React Documentation, Meta Open Source."),
    ])
    return body


def build_docx():
    figures = make_figures()
    image_rels = [(caption, path) for caption, path in figures]
    body = build_body(image_rels)

    doc_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas"
xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"
xmlns:o="urn:schemas-microsoft-com:office:office"
xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"
xmlns:v="urn:schemas-microsoft-com:vml"
xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing"
xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
xmlns:w10="urn:schemas-microsoft-com:office:word"
xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"
xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup"
xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk"
xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml"
xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"
xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"
mc:Ignorable="w14 wp14">
<w:body>
{''.join(body)}
<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr>
</w:body></w:document>'''

    styles_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:after="160" w:line="276" w:lineRule="auto"/></w:pPr><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="24"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:before="240" w:after="240"/></w:pPr><w:rPr><w:b/><w:sz w:val="34"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:before="360" w:after="140"/></w:pPr><w:rPr><w:b/><w:sz w:val="30"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:before="240" w:after="100"/></w:pPr><w:rPr><w:b/><w:sz w:val="26"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="ListParagraph"><w:name w:val="List Paragraph"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:ind w:left="360"/></w:pPr></w:style>
<w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:basedOn w:val="TableNormal"/><w:qFormat/><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4"/><w:left w:val="single" w:sz="4"/><w:bottom w:val="single" w:sz="4"/><w:right w:val="single" w:sz="4"/><w:insideH w:val="single" w:sz="4"/><w:insideV w:val="single" w:sz="4"/></w:tblBorders></w:tblPr></w:style>
</w:styles>'''

    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="png" ContentType="image/png"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>'''

    root_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>'''

    doc_rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
                '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>']
    for i, (_, path) in enumerate(image_rels, start=1):
        doc_rels.append(f'<Relationship Id="rIdImg{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{path.name}"/>')
    doc_rels.append("</Relationships>")
    doc_rels_xml = "\n".join(doc_rels)

    with ZipFile(OUT, "w", ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("word/document.xml", doc_xml)
        zf.writestr("word/styles.xml", styles_xml)
        zf.writestr("word/_rels/document.xml.rels", doc_rels_xml)
        for _, path in image_rels:
            zf.write(path, f"word/media/{path.name}")

    print(f"Created {OUT.resolve()}")
    print(f"Created images in {ASSET_DIR.resolve()}")


if __name__ == "__main__":
    build_docx()
