# Presentation Generation with Marp

This directory contains the presentation generation system for Rhiza.
The project uses [Marp](https://marp.app/) to convert Markdown files into beautiful presentation slides.

## Overview

The presentation system consists of:
- **PRESENTATION.md** — The main presentation source file (located in the repository root)
- **Makefile.presentation** — Make targets for generating and serving presentations (in this directory)
- **Marp CLI** — The tool that converts Markdown to HTML/PDF slides

## Prerequisites

### Required Tools

1. **Node.js and npm** — Required to install Marp CLI
   - Download from: [https://nodejs.org/](https://nodejs.org/)
   - Check installation: `node --version` and `npm --version`

2. **Marp CLI** — The presentation generator
   - The Makefile will automatically install it if not present
   - Manual installation: `npm install -g @marp-team/marp-cli`
   - Check installation: `marp --version`

### Optional Tools

For PDF generation, you may need additional dependencies:
- **Google Chrome/Chromium** — Used by Marp for PDF rendering
- On most systems, this is automatically detected if installed

## Available Commands

The presentation system provides three main commands via the Makefile:

### 1. Generate HTML Presentation

Run from the repository root:

```bash
make presentation
```

This command:
- Checks if Marp CLI is installed (installs it automatically if needed)
- Converts `PRESENTATION.md` to `presentation.html`
- Creates an HTML file that can be opened in any web browser

**Output**: `presentation.html` in the repository root

### 2. Generate PDF Presentation

Run from the repository root:

```bash
make presentation-pdf
```

This command:
- Checks if Marp CLI is installed (installs it automatically if needed)
- Converts `PRESENTATION.md` to `presentation.pdf`
- Creates a PDF file suitable for distribution

**Output**: `presentation.pdf` in the repository root

**Note**: PDF generation requires a Chromium-based browser to be installed.

### 3. Serve Presentation Interactively

Run from the repository root:

```bash
make presentation-serve
```

This command:
- Checks if Marp CLI is installed (installs it automatically if needed)
- Starts a local web server with live reload
- Opens your browser to view the presentation
- Automatically refreshes when you edit `PRESENTATION.md`

**Server**: Usually runs at `http://localhost:8080`

**Stop server**: Press `Ctrl+C` in the terminal

## Creating Your Presentation

### Editing PRESENTATION.md

The source file for your presentation is located at the repository root: `/PRESENTATION.md`

To edit it:

```bash
# Open in your favorite editor
vim PRESENTATION.md
# or
code PRESENTATION.md
# or
nano PRESENTATION.md
```

### Marp Markdown Syntax

Marp extends standard Markdown with special directives for presentations.

#### Basic Structure

```markdown
---
marp: true
theme: default
paginate: true
---

<!-- _class: lead -->
# My First Slide

Content goes here

---

## Second Slide

- Bullet point 1
- Bullet point 2

---

## Third Slide

More content
```

#### Key Directives

- `---` — Creates a new slide
- `<!-- _class: lead -->` — Centers content on the slide
- Front matter (between `---` at the start) — Configures presentation settings

#### Styling

The current presentation uses custom CSS in the front matter:

```yaml
style: |
  section {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  }
  h1 {
    color: #2FA4A9;
  }
```

You can modify these styles to match your branding.

## Common Workflows

### Quick Preview While Editing

For the best editing experience:

1. Open two terminals
2. In terminal 1: `make presentation-serve`
3. In terminal 2: Edit `PRESENTATION.md`
4. Changes appear instantly in your browser

### Generate Final Deliverables

Before presenting or sharing:

```bash
# Generate both HTML and PDF
make presentation
make presentation-pdf
```

This creates:
- `presentation.html` — For web viewing
- `presentation.pdf` — For offline viewing or printing

### Updating the Presentation

1. Edit `PRESENTATION.md` with your changes
2. Regenerate outputs:
   ```bash
   make presentation
   make presentation-pdf
   ```
3. Test in a browser to ensure everything looks correct

## Troubleshooting

### Marp CLI Not Found

**Problem**: `marp: command not found`

**Solution**: The Makefile should install it automatically, but if it doesn't:
```bash
npm install -g @marp-team/marp-cli
```

### npm Not Found

**Problem**: `npm: command not found`

**Solution**: Install Node.js from [https://nodejs.org/](https://nodejs.org/)

### PDF Generation Fails

**Problem**: `Error: Failed to launch browser`

**Solution**: Install Google Chrome or Chromium:
- **Ubuntu/Debian**: `sudo apt-get install chromium-browser`
- **macOS**: `brew install chromium`
- **Windows**: Download from [https://www.google.com/chrome/](https://www.google.com/chrome/)

### Permission Errors During npm Install

**Problem**: `EACCES: permission denied` when installing Marp

**Solution**: Either:
- Use `sudo` (not recommended): `sudo npm install -g @marp-team/marp-cli`
- Configure npm to use a local directory (recommended):
  ```bash
  mkdir ~/.npm-global
  npm config set prefix '~/.npm-global'
  echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
  source ~/.bashrc
  npm install -g @marp-team/marp-cli
  ```

### Styles Not Applying

**Problem**: Custom styles in front matter don't appear

**Solution**:
- Ensure `marp: true` is set in the front matter
- Check that your CSS syntax is valid
- Try clearing browser cache

## Advanced Usage

### Custom Themes

Create a custom Marp theme:

1. Create a CSS file with your theme (e.g., `custom-theme.css`)
2. Reference it in the front matter of `PRESENTATION.md`:
   ```yaml
   ---
   marp: true
   theme: custom-theme
   ---
   ```
3. Modify the Makefile targets to include your theme directory:
   ```makefile
   presentation: ## generate presentation slides with custom theme
   	@marp PRESENTATION.md --theme-set custom-theme.css -o presentation.html
   ```

### Exporting to PowerPoint

While Marp doesn't directly export to PowerPoint, you can:
1. Generate PDF: `make presentation-pdf`
2. Use a PDF-to-PPTX converter online or with Adobe Acrobat

### Multiple Presentations

To create additional presentations:
1. Create a new Markdown file (e.g., `WORKSHOP.md`)
2. Add new targets to `presentation/Makefile.presentation` following the existing pattern:
   ```makefile
   workshop: ## generate workshop slides from WORKSHOP.md using Marp
   	@printf "${BLUE}[INFO] Checking for Marp CLI...${RESET}\n"
   	@if ! command -v marp >/dev/null 2>&1; then \
   	  if command -v npm >/dev/null 2>&1; then \
   	    printf "${YELLOW}[WARN] Marp CLI not found. Installing with npm...${RESET}\n"; \
   	    npm install -g @marp-team/marp-cli || { \
   	      printf "${RED}[ERROR] Failed to install Marp CLI.${RESET}\n"; \
   	      exit 1; \
   	    }; \
   	  else \
   	    printf "${RED}[ERROR] npm not found.${RESET}\n"; \
   	    exit 1; \
   	  fi; \
   	fi
   	@printf "${BLUE}[INFO] Generating HTML workshop slides...${RESET}\n"
   	@marp WORKSHOP.md -o workshop.html
   	@printf "${GREEN}[SUCCESS] Workshop slides generated: workshop.html${RESET}\n"
   ```
3. Run: `make workshop`

## Learn More

- **Marp Documentation**: [https://marpit.marp.app/](https://marpit.marp.app/)
- **Marp CLI Documentation**: [https://github.com/marp-team/marp-cli](https://github.com/marp-team/marp-cli)
- **Marpit Markdown**: [https://marpit.marp.app/markdown](https://marpit.marp.app/markdown)
- **Theme Customization**: [https://marpit.marp.app/theme-css](https://marpit.marp.app/theme-css)

## Integration with Rhiza

This presentation system is part of the Rhiza template collection. When you integrate Rhiza into your project, you automatically get:

- ✅ The Makefile targets for presentation generation
- ✅ A sample `PRESENTATION.md` file
- ✅ Automatic Marp CLI installation
- ✅ GitHub Actions integration (optional)

The presentation targets are included in the main Makefile through:
```makefile
-include presentation/Makefile.presentation
```

## Contributing

If you improve the presentation system:
1. Update `Makefile.presentation` for new features
2. Update this README with documentation
3. Update `PRESENTATION.md` with examples
4. Test all three commands: `presentation`, `presentation-pdf`, `presentation-serve`

## License

This presentation system is part of Rhiza and is licensed under the MIT License.
