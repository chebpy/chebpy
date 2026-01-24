"""Marimo Showcase Notebook - Demonstrating Key Features.

This notebook showcases the most useful features of Marimo, including:
- Interactive UI elements (sliders, dropdowns, text inputs)
- Reactive programming (automatic cell updates)
- Data visualisation with popular libraries
- Markdown and LaTeX support
- Layout components (columns, tabs, accordions)
- Forms and user input handling
- Dynamic content generation

Run this notebook with: marimo edit rhiza.py
Or in the rhiza project: make marimo
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.18.4",
#     "numpy>=1.24.0",
#     "plotly>=5.18.0",
#     "pandas>=2.0.0",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go


@app.cell
def cell_02():
    """Render the showcase introduction Markdown content."""
    mo.md(
        r"""
        # 🎨 Marimo Showcase

        Welcome to the **Marimo Showcase Notebook**! This interactive notebook demonstrates
        the most powerful and useful features of [Marimo](https://marimo.io/).

        **Marimo** is a reactive Python notebook that combines the best of Jupyter notebooks
        with the power of reactive programming. Every cell automatically updates when its
        dependencies change, creating a seamless interactive experience.

        ## Why Marimo?

        - ✨ **Reactive by default** - No manual cell re-runs
        - 🎯 **Pure Python** - Notebooks are `.py` files
        - 🔄 **Reproducible** - Consistent execution order
        - 🎨 **Rich UI elements** - Beautiful interactive components
        - 📦 **Version control friendly** - Easy to diff and merge
        """
    )


@app.cell
def cell_03():
    """Render a horizontal rule to separate sections."""
    mo.md(r"""---""")


@app.cell
def cell_04():
    """Introduce the Interactive UI Elements section."""
    mo.md(
        r"""
        ## 🎚️ Interactive UI Elements

        Marimo provides rich UI components that automatically trigger reactive updates.
        """
    )


@app.cell
def cell_05():
    """Create and display a numeric slider UI component."""
    # Slider for numeric input
    slider = mo.ui.slider(start=0, stop=100, value=50, label="Adjust the value:", show_value=True)
    slider
    return (slider,)


@app.cell
def cell_06(slider):
    """Display the current slider value reactively."""
    mo.md(
        f"""
        The slider value is: **{slider.value}**

        This text updates automatically when you move the slider! ✨
        """
    )


@app.cell
def cell_07():
    """Create and display a dropdown for language selection."""
    # Dropdown for selection
    dropdown = mo.ui.dropdown(
        options=["Python", "JavaScript", "Rust", "Go", "TypeScript"],
        value="Python",
        label="Choose your favorite language:",
    )
    dropdown
    return (dropdown,)


@app.cell
def cell_08(dropdown):
    """Display the currently selected language from the dropdown."""
    mo.md(
        f"""
        You selected: **{dropdown.value}** 🎉

        Great choice! {dropdown.value} is an excellent programming language.
        """
    )


@app.cell
def cell_09():
    """Create and display a text input field for the user's name."""
    # Text input
    text_input = mo.ui.text(value="Marimo", label="Enter your name:", placeholder="Type something...")
    text_input
    return (text_input,)


@app.cell
def cell_10(text_input):
    """Display a personalized greeting using the current text input value."""
    mo.md(f"""Hello, **{text_input.value}**! 👋""")


@app.cell
def cell_11():
    """Render a horizontal rule to visually separate sections."""
    mo.md(r"""---""")


@app.cell
def cell_12():
    """Introduce the Data Visualisation section."""
    mo.md(
        r"""
        ## 📊 Data Visualisation

        Marimo works seamlessly with popular visualisation libraries like Plotly,
        Altair, and Matplotlib. Let's create interactive plots!
        """
    )


@app.cell
def cell_14():
    """Create sliders for wave frequency and amplitude controls for the plot."""
    # Interactive controls for the plot
    frequency_slider = mo.ui.slider(start=1, stop=10, value=2, label="Wave frequency:", show_value=True)

    amplitude_slider = mo.ui.slider(start=1, stop=5, value=1, label="Wave amplitude:", show_value=True)

    mo.vstack([frequency_slider, amplitude_slider])
    return amplitude_slider, frequency_slider


@app.cell
def cell_15(amplitude_slider, frequency_slider):
    """Build a reactive Plotly sine wave based on the slider values."""
    # Generate reactive plot based on slider values
    x = np.linspace(0, 4 * np.pi, 1000)
    y = amplitude_slider.value * np.sin(frequency_slider.value * x)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line={"color": "#2FA4A9", "width": 2}, name="Sine Wave"))

    fig.update_layout(
        title=f"Sine Wave: y = {amplitude_slider.value} × sin({frequency_slider.value}x)",
        xaxis_title="x",
        yaxis_title="y",
        template="plotly_white",
        height=400,
        showlegend=False,
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")

    mo.vstack(
        [
            mo.md(
                f"""
            ### Interactive Sine Wave

            Adjust the sliders above to change the wave properties!

            Current parameters:
            - Frequency: {frequency_slider.value}
            - Amplitude: {amplitude_slider.value}
            """
            ),
            mo.ui.plotly(fig),
        ]
    )
    return fig, x, y


@app.cell
def cell_16():
    """Render a horizontal rule to visually separate sections."""
    mo.md(r"""---""")


@app.cell
def cell_17():
    """Introduce the DataFrames section."""
    mo.md(
        r"""
        ## 📋 Working with DataFrames

        Marimo provides excellent support for working with Pandas DataFrames.
        """
    )


@app.cell
def cell_18():
    """Create a sample Pandas DataFrame for use in subsequent cells."""
    # Create sample data
    data = pd.DataFrame(
        {
            "Product": ["Widget A", "Widget B", "Widget C", "Widget D", "Widget E"],
            "Sales": [250, 180, 420, 350, 290],
            "Revenue": [5000, 3600, 8400, 7000, 5800],
            "Rating": [4.5, 4.2, 4.8, 4.6, 4.3],
        }
    )
    return data


@app.cell
def cell_19():
    """Render introductory text for the sample sales dataset."""
    mo.md(
        r"""
        ### Sample Sales Data

        Here's our dataset displayed as an interactive table:
        """
    )


@app.cell
def cell_20(data):
    """Display the sample dataset as an interactive table."""
    # Display as interactive table
    mo.ui.table(data)


@app.cell
def cell_21(data):
    """Render a Plotly bar chart showing sales by product."""
    # Create a bar chart with Plotly
    colours = ["#2FA4A9", "#3FB5BA", "#4FC6CB", "#5FD7DC", "#6FE8ED"]

    fig_bar = go.Figure()
    fig_bar.add_trace(
        go.Bar(
            x=data["Product"],
            y=data["Sales"],
            marker_color=colours,
            text=data["Sales"],
            textposition="auto",
        )
    )

    fig_bar.update_layout(
        title="Sales by Product",
        xaxis_title="Product",
        yaxis_title="Sales",
        template="plotly_white",
        height=500,
        showlegend=False,
    )

    mo.ui.plotly(fig_bar)
    return colours, fig_bar


@app.cell
def cell_22():
    """Render a horizontal rule to visually separate sections."""
    mo.md(r"""---""")


@app.cell
def cell_23():
    """Introduce the layout components section."""
    mo.md(
        r"""
        ## 🎯 Layout Components

        Marimo provides powerful layout primitives to organise your content.
        """
    )


@app.cell
def cell_24():
    """Demonstrate a two-column layout with left and right content."""
    # Using columns for side-by-side layout
    left_content = mo.md(
        r"""
        ### Left Column

        This is the left side of a two-column layout.

        - Feature 1
        - Feature 2
        - Feature 3
        """
    )

    right_content = mo.md(
        r"""
        ### Right Column

        This is the right side of a two-column layout.

        - Benefit A
        - Benefit B
        - Benefit C
        """
    )

    mo.hstack([left_content, right_content], justify="space-between")
    return left_content, right_content


@app.cell
def cell_25():
    """Demonstrate tabs with Introduction, Details, and Summary content."""
    # Using tabs for organised content
    tab1 = mo.md(
        r"""
        ## Tab 1: Introduction

        This is the content of the first tab. Tabs are great for organising
        related content without cluttering the interface.

        **Key points:**
        - Clean organisation
        - Reduced clutter
        - Easy navigation
        """
    )

    tab2 = mo.md(
        r"""
        ## Tab 2: Details

        Here's more detailed information in the second tab.

        You can include any content here:
        - Code examples
        - Visualisations
        - Interactive elements
        """
    )

    tab3 = mo.md(
        r"""
        ## Tab 3: Summary

        The final tab with summary information.

        Tabs are perfect for:
        1. Step-by-step guides
        2. Different views of data
        3. Organising complex notebooks
        """
    )

    mo.ui.tabs({"Introduction": tab1, "Details": tab2, "Summary": tab3})
    return tab1, tab2, tab3


@app.cell
def cell_26():
    """Render a horizontal rule to visually separate sections."""
    mo.md(r"""---""")


@app.cell
def cell_27():
    """Introduce the forms and user input section."""
    mo.md(
        r"""
        ## 📝 Forms and User Input

        Marimo forms allow you to batch multiple inputs and submit them together.
        """
    )


@app.cell
def cell_28():
    """Build and display a multi-input form for collecting user information."""
    # Create a form with multiple inputs
    form = mo.ui.dictionary(
        {
            "name": mo.ui.text(label="Your name:", placeholder="Enter name"),
            "age": mo.ui.slider(start=18, stop=100, value=25, label="Your age:"),
            "email": mo.ui.text(label="Email:", placeholder="email@example.com"),
            "subscribe": mo.ui.checkbox(label="Subscribe to newsletter"),
            "interests": mo.ui.multiselect(
                options=["Data Science", "Machine Learning", "Web Development", "DevOps"], label="Your interests:"
            ),
        }
    )
    mo.vstack([mo.md("### User Information Form"), form])
    return (form,)


@app.cell
def cell_29(form):
    """Display current form values reactively as the user edits the form."""
    # Display form values - updates reactively as you type/change values
    if form.value and any(form.value.values()):
        interests_text = ", ".join(form.value["interests"]) if form.value["interests"] else "None selected"

        mo.md(
            f"""
            ### Current Form Values ✅

            The values update automatically as you interact with the form!

            - **Name:** {form.value["name"] or "(not entered)"}
            - **Age:** {form.value["age"]}
            - **Email:** {form.value["email"] or "(not entered)"}
            - **Newsletter:** {"Subscribed ✅" if form.value["subscribe"] else "Not subscribed"}
            - **Interests:** {interests_text}

            Notice how the values update reactively as you change them!
            """
        )
    else:
        mo.md("*The form values will appear here as you interact with them.*")


@app.cell
def cell_30():
    """Render a horizontal rule to visually separate sections."""
    mo.md(r"""---""")


@app.cell
def cell_31():
    """Introduce the Markdown and LaTeX support section."""
    mo.md(
        r"""
        ## 🎓 Markdown & LaTeX Support

        Marimo has excellent support for rich text formatting using Markdown and LaTeX.
        """
    )


@app.cell
def cell_32():
    """Render rich Markdown with LaTeX equations, code blocks, and formatting examples."""
    mo.md(
        r"""
        ### Mathematical Equations

        You can write beautiful equations using LaTeX:

        **Pythagorean theorem:**

        $$a^2 + b^2 = c^2$$

        **Euler's identity:**

        $$e^{i\pi} + 1 = 0$$

        **Quadratic formula:**

        $$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$

        **Inline math:** The famous $E = mc^2$ equation by Einstein.

        ### Code Blocks

        You can also include syntax-highlighted code:

        ```python
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        ```

        ### Rich Formatting

        - **Bold text**
        - *Italic text*
        - ~~Strikethrough text~~
        - `Inline code`
        - [Links](https://marimo.io/)

        > This is a blockquote with important information!
        """
    )


@app.cell
def cell_33():
    """Render a horizontal rule to visually separate sections."""
    mo.md(r"""---""")


@app.cell
def cell_34():
    """Introduce the Advanced Features section."""
    mo.md(
        r"""
        ## 🎪 Advanced Features

        Here are some more advanced Marimo features worth exploring.
        """
    )


@app.cell
def cell_35():
    """Render an informational callout about Marimo notebooks being plain Python files."""
    # Callout boxes for important information
    mo.callout(
        mo.md(
            r"""
            ### 💡 Pro Tip

            Marimo notebooks are **just Python files**! This means:
            - Easy version control with Git
            - Standard code review workflows
            - No hidden JSON metadata
            - Compatible with all Python tools
            """
        ),
        kind="info",
    )


@app.cell
def cell_36():
    """Display an accordion with notes on reactivity, performance, and dependencies."""
    # Accordion for collapsible content
    mo.accordion(
        {
            "🔍 Click to learn about Reactive Programming": mo.md(
                r"""
            Marimo uses **reactive programming** to automatically track dependencies
            between cells. When you change a value in one cell, all dependent cells
            automatically update!

            This eliminates the common notebook problem of running cells out of order.
            """
            ),
            "🚀 Click to learn about Performance": mo.md(
                r"""
            Marimo only re-runs cells that are affected by changes, making it
            efficient even for large notebooks. This intelligent execution means
            you get fast feedback without wasting computation.
            """
            ),
            "📦 Click to learn about Dependencies": mo.md(
                r"""
            You can specify dependencies right in the notebook using inline metadata.
            This makes notebooks self-contained and reproducible, as seen in the
            header of this notebook!
            """
            ),
        }
    )


@app.cell
def cell_37():
    """Render a horizontal rule to visually separate sections."""
    mo.md(r"""---""")


@app.cell
def cell_38():
    """Render the conclusion section of the Marimo showcase notebook."""
    mo.md(
        r"""
        ## 🎉 Conclusion

        This notebook has demonstrated many of Marimo's most useful features:

        ✅ **Interactive UI elements** - Sliders, dropdowns, text inputs, and more
        ✅ **Reactive programming** - Automatic cell updates when dependencies change
        ✅ **Data visualisation** - Seamless integration with Plotly, Matplotlib, etc.
        ✅ **Layout components** - Columns, tabs, accordions for organising content
        ✅ **Forms** - Batched input collection with submission
        ✅ **Rich formatting** - Markdown and LaTeX support
        ✅ **Pure Python** - Notebooks are version-control friendly `.py` files

        ### Next Steps

        To learn more about Marimo:
        - Visit the [official documentation](https://docs.marimo.io/)
        - Explore the [example gallery](https://marimo.io/examples)
        - Join the [community Discord](https://discord.gg/JE7nhX6mD8)

        **Happy exploring! 🚀**
        """
    )


if __name__ == "__main__":
    app.run()
