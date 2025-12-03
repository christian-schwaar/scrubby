/** @type {import('tailwindcss').Config} */
module.exports = {
    darkMode: ["class"],
    content: ["./index.html", "./src/**/*.{ts,tsx}"],
    theme: {
      container: { center: true, padding: "2rem" },
      extend: {
        borderRadius: { lg: "0.5rem", xl: "0.75rem", "2xl": "1rem" },
      },
    },
    plugins: [require("tailwindcss-animate")],
  }