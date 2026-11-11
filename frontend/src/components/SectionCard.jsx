export default function SectionCard({ title, children, className = "" }) {
  return (
    <section className={`section-card ${className}`}>
      <div className="section-header">
        <h2>{title}</h2>
      </div>
      {children}
    </section>
  );
}