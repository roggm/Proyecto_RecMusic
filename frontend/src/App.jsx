import { useState, useEffect } from "react";

const API_BASE = "http://localhost:8000";

const GENEROS = [
  "acoustic","afrobeat","alt-rock","alternative","ambient","anime",
  "black-metal","bluegrass","blues","breakbeat","chill","classical",
  "club","country","dance","dancehall","death-metal","deep-house",
  "disco","drum-and-bass","dub","dubstep","edm","electro","electronic",
  "emo","folk","funk","gospel","grunge","hard-rock","hardcore",
  "heavy-metal","hip-hop","house","indie","indie-pop","jazz","k-pop",
  "latin","metal","metalcore","opera","piano","pop","punk","punk-rock",
  "r-n-b","reggae","reggaeton","rock","rock-n-roll","salsa","singer-songwriter",
  "ska","soul","synth-pop","techno","trance","trip-hop",
];

const FEATURES = [
  { key:"danceability",    label:"¿Qué tanto te gusta bailar?",   izq:"Nada en absoluto",       der:"Todo el tiempo",         icon:"ti-music"       },
  { key:"energy",          label:"¿Cómo prefieres tu música?",    izq:"Tranquila y relajante",  der:"Intensa y energética",   icon:"ti-bolt"        },
  { key:"valence",         label:"¿Qué estado de ánimo buscas?",  izq:"Melancólica / reflexiva",der:"Alegre / positiva",      icon:"ti-mood-smile"  },
  { key:"acousticness",    label:"¿Qué sonido prefieres?",        izq:"Electrónico / producido",der:"Acústico / orgánico",    icon:"ti-guitar-pick" },
  { key:"instrumentalness",label:"¿Letra o instrumental?",        izq:"Con letra (vocal)",      der:"Solo instrumental",      icon:"ti-microphone"  },
];

const PASOS = ["Sobre ti", "Géneros", "Tu sonido", "Resultados"];

function Boton({ onClick, disabled, children, secondary }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        padding: "10px 24px",
        borderRadius: "var(--border-radius-md)",
        border: secondary ? "1px solid var(--color-border-secondary)" : "none",
        background: secondary ? "transparent" : "var(--color-text-info)",
        color: secondary ? "var(--color-text-primary)" : "#fff",
        fontSize: "14px",
        fontWeight: "500",
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.45 : 1,
        transition: "opacity 0.15s",
      }}
    >
      {children}
    </button>
  );
}

function Paso1({ datos, setDatos }) {
  return (
    <div>
      <h2 style={{ fontSize: "22px", fontWeight: "500", marginBottom: "6px" }}>
        Cuéntanos sobre ti
      </h2>
      <p style={{ fontSize: "14px", color: "var(--color-text-secondary)", marginBottom: "2rem" }}>
        Usaremos esta info para personalizar tus recomendaciones.
      </p>
      <div style={{ marginBottom: "1.5rem" }}>
        <label style={{ fontSize: "13px", color: "var(--color-text-secondary)", display: "block", marginBottom: "6px" }}>
          Tu nombre
        </label>
        <input
          type="text"
          placeholder="¿Cómo te llamas?"
          value={datos.nombre}
          onChange={e => setDatos({ ...datos, nombre: e.target.value })}
          style={{ width: "100%", boxSizing: "border-box" }}
        />
      </div>
      <div>
        <label style={{ fontSize: "13px", color: "var(--color-text-secondary)", display: "block", marginBottom: "6px" }}>
          Tu edad
        </label>
        <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
          <input
            type="range" min="13" max="70" step="1"
            value={datos.edad}
            onChange={e => setDatos({ ...datos, edad: parseInt(e.target.value) })}
            style={{ flex: 1 }}
          />
          <span style={{ fontSize: "22px", fontWeight: "500", minWidth: "40px", textAlign: "center" }}>
            {datos.edad}
          </span>
        </div>
      </div>
    </div>
  );
}

function Paso2({ datos, setDatos }) {
  const toggle = (g) => {
    const set = new Set(datos.generos);
    set.has(g) ? set.delete(g) : set.add(g);
    setDatos({ ...datos, generos: Array.from(set) });
  };
  return (
    <div>
      <h2 style={{ fontSize: "22px", fontWeight: "500", marginBottom: "6px" }}>
        ¿Qué géneros te gustan?
      </h2>
      <p style={{ fontSize: "14px", color: "var(--color-text-secondary)", marginBottom: "1.5rem" }}>
        Selecciona todos los que escuches. Mínimo 1.
      </p>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
        {GENEROS.map(g => {
          const sel = datos.generos.includes(g);
          return (
            <button key={g} onClick={() => toggle(g)} style={{
              padding: "6px 14px",
              borderRadius: "var(--border-radius-md)",
              border: sel ? "1.5px solid var(--color-text-info)" : "1px solid var(--color-border-secondary)",
              background: sel ? "var(--color-background-info)" : "var(--color-background-primary)",
              color: sel ? "var(--color-text-info)" : "var(--color-text-primary)",
              fontSize: "13px", cursor: "pointer",
              fontWeight: sel ? "500" : "400", transition: "all 0.15s",
            }}>
              {g}
            </button>
          );
        })}
      </div>
      {datos.generos.length > 0 && (
        <p style={{ fontSize: "12px", color: "var(--color-text-tertiary)", marginTop: "1rem" }}>
          {datos.generos.length} géneros seleccionados
        </p>
      )}
    </div>
  );
}

function Paso3({ datos, setDatos }) {
  return (
    <div>
      <h2 style={{ fontSize: "22px", fontWeight: "500", marginBottom: "6px" }}>
        ¿Cómo es tu música ideal?
      </h2>
      <p style={{ fontSize: "14px", color: "var(--color-text-secondary)", marginBottom: "2rem" }}>
        Mueve los sliders según tu preferencia.
      </p>
      {FEATURES.map(f => (
        <div key={f.key} style={{
          marginBottom: "1.75rem", padding: "1rem 1.25rem",
          background: "var(--color-background-primary)",
          border: "1px solid var(--color-border-tertiary)",
          borderRadius: "var(--border-radius-lg)",
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "12px" }}>
            <i className={`ti ${f.icon}`} style={{ fontSize: "16px", color: "var(--color-text-info)" }} />
            <span style={{ fontSize: "14px", fontWeight: "500" }}>{f.label}</span>
          </div>
          <input
            type="range" min="0" max="100" step="1"
            value={Math.round(datos.features[f.key] * 100)}
            onChange={e => setDatos({ ...datos, features: { ...datos.features, [f.key]: parseInt(e.target.value) / 100 } })}
            style={{ width: "100%", marginBottom: "6px" }}
          />
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <span style={{ fontSize: "11px", color: "var(--color-text-tertiary)" }}>{f.izq}</span>
            <span style={{ fontSize: "11px", color: "var(--color-text-tertiary)" }}>{f.der}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

function TarjetaCancion({ rec, index }) {
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: "14px",
      padding: "12px 16px",
      background: "var(--color-background-primary)",
      border: "1px solid var(--color-border-tertiary)",
      borderRadius: "var(--border-radius-lg)",
    }}>
      <span style={{ fontSize: "13px", fontWeight: "500", color: "var(--color-text-tertiary)", minWidth: "20px" }}>
        {index + 1}
      </span>
      <div style={{
        width: "36px", height: "36px", borderRadius: "var(--border-radius-md)",
        background: "var(--color-background-info)",
        display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
      }}>
        <i className="ti ti-music" style={{ fontSize: "16px", color: "var(--color-text-info)" }} />
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <p style={{ margin: 0, fontSize: "14px", fontWeight: "500", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
          {rec.titulo || rec.item_id}
        </p>
        <p style={{ margin: 0, fontSize: "12px", color: "var(--color-text-secondary)" }}>
          {rec.artista} {rec.genero && `· ${rec.genero}`}
        </p>
      </div>
      {rec.popularidad > 0 && (
        <span style={{
          fontSize: "11px", padding: "3px 8px",
          borderRadius: "var(--border-radius-md)",
          background: "var(--color-background-secondary)",
          color: "var(--color-text-secondary)", flexShrink: 0,
        }}>
          {Math.round(rec.popularidad)}% pop
        </span>
      )}
    </div>
  );
}

function PasoResultados({ datos }) {
  const [estado, setEstado] = useState("cargando");
  const [recomendaciones, setRecomendaciones] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchRecs = async () => {
      try {
        const res = await fetch(`${API_BASE}/recommendations/new-user`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_id: datos.nombre.trim(),
            generos: datos.generos,
            features: datos.features,
            cancion_referencia_id: null,
          }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        setRecomendaciones(json.recomendaciones || []);
        setEstado("listo");
      } catch (e) {
        setError(e.message);
        setEstado("error");
      }
    };
    fetchRecs();
  }, []);

  if (estado === "cargando") {
    return (
      <div style={{ textAlign: "center", padding: "3rem 0" }}>
        <i className="ti ti-loader" style={{ fontSize: "32px", color: "var(--color-text-info)", marginBottom: "1rem", display: "block" }} />
        <p style={{ fontSize: "15px", color: "var(--color-text-secondary)" }}>
          Buscando canciones para {datos.nombre || "ti"}...
        </p>
      </div>
    );
  }

  if (estado === "error") {
    return (
      <div style={{ padding: "1.5rem", background: "var(--color-background-danger)", borderRadius: "var(--border-radius-lg)" }}>
        <p style={{ fontSize: "14px", color: "var(--color-text-danger)", margin: 0 }}>
          <i className="ti ti-alert-circle" style={{ marginRight: "6px" }} />
          {error || "No se pudieron cargar las recomendaciones"}
        </p>
      </div>
    );
  }

  return (
    <div>
      <h2 style={{ fontSize: "22px", fontWeight: "500", marginBottom: "4px" }}>
        Tu playlist personalizada 🎧
      </h2>
      <p style={{ fontSize: "14px", color: "var(--color-text-secondary)", marginBottom: "1.5rem" }}>
        Basada en tus géneros y perfil de audio, {datos.nombre || "aquí va tu recomendación"}.
      </p>
      <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
        {recomendaciones.map((rec, i) => (
          <TarjetaCancion key={rec.item_id || i} rec={rec} index={i} />
        ))}
      </div>
      {recomendaciones.length === 0 && (
        <p style={{ fontSize: "14px", color: "var(--color-text-tertiary)", textAlign: "center", padding: "2rem 0" }}>
          No se encontraron recomendaciones para este perfil.
        </p>
      )}
    </div>
  );
}

export default function App() {
  const [paso, setPaso] = useState(0);
  const [datos, setDatos] = useState({
    nombre: "",
    edad: 22,
    generos: [],
    features: {
      danceability: 0.5, energy: 0.5, valence: 0.5,
      acousticness: 0.5, instrumentalness: 0.2,
    },
  });

  const puedeAvanzar = () => {
    if (paso === 0) return datos.nombre.trim().length > 0;
    if (paso === 1) return datos.generos.length > 0;
    return true;
  };

  const avanzar = () => setPaso(p => p + 1);
  const retroceder = () => setPaso(p => p - 1);
  const TOTAL_PASOS = PASOS.length;

  return (
    <div style={{ maxWidth: "600px", margin: "0 auto", padding: "2rem 1rem" }}>

      <div style={{ marginBottom: "2rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "4px" }}>
          <i className="ti ti-headphones" style={{ fontSize: "22px", color: "var(--color-text-info)" }} />
          <span style={{ fontSize: "16px", fontWeight: "500" }}>RecMusic</span>
        </div>
        <p style={{ fontSize: "13px", color: "var(--color-text-tertiary)", margin: 0 }}>
          Sistema recomendador de música
        </p>
      </div>

      <div style={{
        background: "var(--color-background-primary)",
        border: "1px solid var(--color-border-tertiary)",
        borderRadius: "var(--border-radius-lg)",
        padding: "1.5rem",
        marginBottom: "1.5rem",
        minHeight: "300px",
      }}>
        {paso === 0 && <Paso1 datos={datos} setDatos={setDatos} />}
        {paso === 1 && <Paso2 datos={datos} setDatos={setDatos} />}
        {paso === 2 && <Paso3 datos={datos} setDatos={setDatos} />}
        {paso === 3 && <PasoResultados datos={datos} />}
      </div>

      {paso < 3 && (
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <Boton onClick={retroceder} disabled={paso === 0} secondary>
            ← Atrás
          </Boton>
          <span style={{ fontSize: "12px", color: "var(--color-text-tertiary)" }}>
            {paso + 1} / {TOTAL_PASOS}
          </span>
          <Boton onClick={avanzar} disabled={!puedeAvanzar()}>
            {paso === 2 ? "Ver mis recomendaciones →" : "Siguiente →"}
          </Boton>
        </div>
      )}
    </div>
  );
}
