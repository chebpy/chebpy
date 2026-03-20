# Project Book and Documentation

This directory contains the source and templates for generating the Rhiza companion book and API documentation.

## Structure

- `marimo/`: Interactive [Marimo](https://marimo.io/) notebooks that are included in the book.
- `minibook-templates/`: Jinja2 templates for the minibook generation.
- `pdoc-templates/`: Custom templates for [pdoc](https://pdoc.dev/) API documentation.
- `book.mk`: Specialised Makefile for building the book and documentation.

## Building the Book

You can build the complete documentation book using the main project Makefile:

```bash
make book
```

This process involves:
1. Exporting Marimo notebooks to HTML.
2. Generating API documentation from the source code.
3. Combining them into a cohesive "book" structure.

## Documentation Customisation

You can customise the look and feel of your documentation by providing your own templates.

### API Documentation (pdoc)

The `make docs` command checks for a directory at `book/pdoc-templates`. If found, it uses the templates within that directory to generate the API documentation.

To customise the API docs:
1. Create the directory: `mkdir -p book/pdoc-templates`
2. Add your Jinja2 templates (e.g., `module.html.jinja2`) to this directory.

See the [pdoc documentation](https://pdoc.dev/docs/pdoc.html#templates) for more details on templating.

### Project Logo

The documentation generation supports embedding a project logo in the sidebar.

**Default Behavior:**
By default, the build looks for `assets/rhiza-logo.svg`.

**Customization:**
You can change the logo by setting the `LOGO_FILE` variable in your project's `Makefile` or `local.mk`.

```makefile
# Example: Use a custom PNG logo
LOGO_FILE := assets/my-company-logo.png
```

To disable the logo entirely, set the variable to an empty string:

```makefile
# Example: Disable logo
LOGO_FILE :=
```

### Companion Book (minibook)

The `make book` command checks for a template at `book/minibook-templates/custom.html.jinja2`. If found, it uses this template for the minibook generation.

To customise the book:
1. Create the directory: `mkdir -p book/minibook-templates`
2. Create your custom template at `book/minibook-templates/custom.html.jinja2`.
