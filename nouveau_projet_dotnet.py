import os
import re
import json
import secrets
import requests
import subprocess
from pathlib import Path

PROJECTS_DIR = Path.home() / "Documents" / "Projects"
DOTNET_VERSION = "9.0"

def run(cmd, cwd=None):
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "(aucun message)"
        print(f"❌ Erreur lors de : {cmd}")
        print(f"   {detail}")
        exit(1)
    return result.stdout.strip()

print("=== Nouveau projet .NET 9 ===\n")

# === 1. Nom du projet ===
nom = input("Nom du projet : ").strip()
if not nom:
    print("❌ Nom invalide.")
    exit(1)
if nom[0].isdigit():
    print("❌ Le nom du projet ne peut pas commencer par un chiffre (invalide en C#).")
    exit(1)
if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', nom):
    print("❌ Le nom doit contenir uniquement des lettres, chiffres et underscores.")
    exit(1)

# === 2. Type de projet ===
print("\nType de projet :")
print("  1 - Web App (MVC)")
print("  2 - API REST (minimal API)")
print("  3 - Console / Terminal")
choix_type = input("Ton choix (1 / 2 / 3) : ").strip()
types = {"1": ("webapp", "Web App MVC"), "2": ("webapi", "API REST"), "3": ("console", "Console")}
if choix_type not in types:
    print("❌ Choix invalide.")
    exit(1)
template, label_type = types[choix_type]

# === 3. Base de données ===
print("\nBase de données :")
print("  1 - SQLite   (simple, fichier local, idéal pour débuter)")
print("  2 - PostgreSQL")
print("  3 - SQL Server")
choix_db = input("Ton choix (1 / 2 / 3) : ").strip()
dbs = {
    "1": ("sqlite",     "Microsoft.EntityFrameworkCore.Sqlite",     "Data Source={nom}.db"),
    "2": ("postgresql", "Npgsql.EntityFrameworkCore.PostgreSQL",     "Host=localhost;Database={nom};Username=postgres;Password=CHANGE_MOI"),
    "3": ("sqlserver",  "Microsoft.EntityFrameworkCore.SqlServer",   "Server=localhost;Database={nom};Trusted_Connection=True;"),
}
if choix_db not in dbs:
    print("❌ Choix invalide.")
    exit(1)
db_type, ef_package, conn_template = dbs[choix_db]
conn_string = conn_template.replace("{nom}", nom)

# === 4. Créer le dossier solution ===
solution_dir = PROJECTS_DIR / nom
if solution_dir.exists():
    print(f"\n❌ Le dossier {solution_dir} existe déjà.")
    exit(1)

solution_dir.mkdir(parents=True)
print(f"\n📁 Dossier créé : {solution_dir}")

# === 5. Créer le fichier global.json pour forcer .NET 9 ===
global_json = solution_dir / "global.json"
global_json.write_text(f'{{\n  "sdk": {{\n    "version": "{DOTNET_VERSION}",\n    "rollForward": "latestMinor"\n  }}\n}}\n')

# === 6. Créer la solution ===
print(f"⚙️  Création de la solution {nom}...")
run(f"dotnet new sln --name {nom}", cwd=solution_dir)

# === 7. Créer le projet principal ===
projet_dir = solution_dir / nom
print(f"⚙️  Création du projet {label_type}...")
run(f"dotnet new {template} --name {nom} --framework net9.0", cwd=solution_dir)
run(f"dotnet sln add {nom}/{nom}.csproj", cwd=solution_dir)

# === 8. Ajouter Entity Framework ===
EF_VERSION = "9.*"
print(f"📦 Installation d'Entity Framework ({db_type})...")
run(f"dotnet add package {ef_package} --version \"{EF_VERSION}\"", cwd=projet_dir)
run(f"dotnet add package Microsoft.EntityFrameworkCore.Design --version \"{EF_VERSION}\"", cwd=projet_dir)
run(f"dotnet tool install --global dotnet-ef --version \"{EF_VERSION}\" 2>/dev/null; true", cwd=projet_dir)

# === 9. Créer le DbContext ===
data_dir = projet_dir / "Data"
data_dir.mkdir()

db_context = f"""using Microsoft.EntityFrameworkCore;

namespace {nom}.Data;

public class {nom}DbContext : DbContext
{{
    public {nom}DbContext(DbContextOptions<{nom}DbContext> options) : base(options) {{ }}

    // Ajoute tes DbSet ici
    // Exemple : public DbSet<MonModele> MonModele {{ get; set; }}
}}
"""
(data_dir / f"{nom}DbContext.cs").write_text(db_context)
print(f"✅ DbContext créé : Data/{nom}DbContext.cs")

# === 10. Configurer la connection string + JWT dans appsettings.json ===
jwt_secret = secrets.token_hex(32)
appsettings_path = projet_dir / "appsettings.json"

conn_block = f'"ConnectionStrings": {{\n    "Default": "{conn_string}"\n  }}'

if template in ("webapp", "webapi"):
    jwt_block = f'"Jwt": {{\n    "SecretKey": "{jwt_secret}",\n    "Issuer": "{nom}",\n    "Audience": "{nom}"\n  }}'
    extra = f",\n  {conn_block},\n  {jwt_block}"
else:
    extra = f",\n  {conn_block}"

if appsettings_path.exists():
    contenu = appsettings_path.read_text()
    contenu = contenu.replace('"AllowedHosts": "*"', f'"AllowedHosts": "*"{extra}')
    appsettings_path.write_text(contenu)
else:
    appsettings_path.write_text(f'{{\n  "AllowedHosts": "*"{extra}\n}}\n')
print("✅ Connection string ajoutée dans appsettings.json")
if template in ("webapp", "webapi"):
    print("✅ Clé JWT générée dans appsettings.json")

# === 11. Enregistrer le DbContext + JWT dans Program.cs (webapp et webapi seulement) ===
program_path = projet_dir / "Program.cs"
if template in ("webapp", "webapi") and program_path.exists():
    # Installer les packages JWT + Swagger
    print("📦 Installation de JWT Bearer + Swagger...")
    run(f"dotnet add package Microsoft.AspNetCore.Authentication.JwtBearer --version \"{EF_VERSION}\"", cwd=projet_dir)
    run("dotnet add package BCrypt.Net-Next", cwd=projet_dir)
    run("dotnet add package Swashbuckle.AspNetCore --version \"7.*\"", cwd=projet_dir)

    if db_type == "sqlite":
        db_line = f'builder.Services.AddDbContext<{nom}DbContext>(opt => opt.UseSqlite(builder.Configuration.GetConnectionString("Default")));'
    elif db_type == "postgresql":
        db_line = f'builder.Services.AddDbContext<{nom}DbContext>(opt => opt.UseNpgsql(builder.Configuration.GetConnectionString("Default")));'
    else:
        db_line = f'builder.Services.AddDbContext<{nom}DbContext>(opt => opt.UseSqlServer(builder.Configuration.GetConnectionString("Default")));'

    if template == "webapi":
        # Réécriture complète pour webapi avec Swagger + JWT
        program_path.write_text(f"""using {nom}.Data;
using Microsoft.EntityFrameworkCore;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.IdentityModel.Tokens;
using Microsoft.OpenApi.Models;
using Swashbuckle.AspNetCore.SwaggerGen;
using System.Text;

var builder = WebApplication.CreateBuilder(args);

{db_line}

var jwtKey = builder.Configuration["Jwt:SecretKey"]!;
builder.Services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
    .AddJwtBearer(options =>
    {{
        options.TokenValidationParameters = new TokenValidationParameters
        {{
            ValidateIssuer = true,
            ValidateAudience = true,
            ValidateLifetime = true,
            ValidateIssuerSigningKey = true,
            ValidIssuer = builder.Configuration["Jwt:Issuer"],
            ValidAudience = builder.Configuration["Jwt:Audience"],
            IssuerSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(jwtKey))
        }};
    }});
builder.Services.AddAuthorization();
builder.Services.AddControllers(options =>
{{
    // Tous les endpoints sont protégés par défaut
    var policy = new Microsoft.AspNetCore.Authorization.AuthorizationPolicyBuilder()
        .RequireAuthenticatedUser()
        .Build();
    options.Filters.Add(new Microsoft.AspNetCore.Mvc.Authorization.AuthorizeFilter(policy));
}});

builder.Services.AddSwaggerGen(c =>
{{
    c.SwaggerDoc("v1", new OpenApiInfo {{ Title = "{nom} API", Version = "v1" }});
    c.AddSecurityDefinition("Bearer", new OpenApiSecurityScheme
    {{
        Name = "Authorization",
        Type = SecuritySchemeType.Http,
        Scheme = "Bearer",
        BearerFormat = "JWT",
        In = ParameterLocation.Header,
        Description = "Entre ton token JWT : Bearer {{token}}"
    }});
    c.AddSecurityRequirement(new OpenApiSecurityRequirement
    {{
        {{
            new OpenApiSecurityScheme
            {{
                Reference = new OpenApiReference {{ Type = ReferenceType.SecurityScheme, Id = "Bearer" }}
            }},
            Array.Empty<string>()
        }}
    }});
}});

var app = builder.Build();

if (app.Environment.IsDevelopment())
{{
    app.UseSwagger();
    app.UseSwaggerUI(c => c.SwaggerEndpoint("/swagger/v1/swagger.json", "{nom} API v1"));
}}

app.UseHttpsRedirection();
app.UseAuthentication();
app.UseAuthorization();
app.MapControllers();

// Appliquer les migrations automatiquement au démarrage
using (var scope = app.Services.CreateScope())
{{
    var db = scope.ServiceProvider.GetRequiredService<{nom}DbContext>();
    db.Database.Migrate();
}}

app.Run();
""")
    else:
        # webapp (MVC) — modification du Program.cs existant
        contenu = program_path.read_text()
        injection = f"using {nom}.Data;\nusing Microsoft.EntityFrameworkCore;\nusing Microsoft.AspNetCore.Authentication.JwtBearer;\nusing Microsoft.IdentityModel.Tokens;\nusing System.Text;\n\n"
        jwt_block = f"""
var jwtKey = builder.Configuration["Jwt:SecretKey"]!;
builder.Services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
    .AddJwtBearer(options =>
    {{
        options.TokenValidationParameters = new TokenValidationParameters
        {{
            ValidateIssuer = true,
            ValidateAudience = true,
            ValidateLifetime = true,
            ValidateIssuerSigningKey = true,
            ValidIssuer = builder.Configuration["Jwt:Issuer"],
            ValidAudience = builder.Configuration["Jwt:Audience"],
            IssuerSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(jwtKey))
        }};
    }});
builder.Services.AddAuthorization();
"""
        contenu = injection + contenu.replace(
            "var builder = WebApplication.CreateBuilder(args);",
            f"var builder = WebApplication.CreateBuilder(args);\n\n{db_line}\n{jwt_block}"
        ).replace(
            "app.UseAuthorization();",
            "app.UseAuthentication();\napp.UseAuthorization();"
        )
        program_path.write_text(contenu)

    print("✅ DbContext + JWT enregistrés dans Program.cs")

    # === Créer le modèle User ===
    models_dir = projet_dir / "Models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "User.cs").write_text(f"""namespace {nom}.Models;

public class User
{{
    public int Id {{ get; set; }}
    public string Username {{ get; set; }} = string.Empty;
    public string Email {{ get; set; }} = string.Empty;
    public string PasswordHash {{ get; set; }} = string.Empty;
    public DateTime CreatedAt {{ get; set; }} = DateTime.UtcNow;
}}
""")
    print("✅ Modèle User créé : Models/User.cs")

    # === Réécrire le DbContext avec User ===
    db_context_path = data_dir / f"{nom}DbContext.cs"
    db_context_path.write_text(f"""using Microsoft.EntityFrameworkCore;
using {nom}.Models;

namespace {nom}.Data;

public class {nom}DbContext : DbContext
{{
    public {nom}DbContext(DbContextOptions<{nom}DbContext> options) : base(options) {{ }}

    public DbSet<User> Users {{ get; set; }}

    // Ajoute tes autres DbSet ici
}}
""")
    print("✅ DbSet<User> ajouté au DbContext")

    # === Créer les DTOs Register / Login ===
    dtos_dir = projet_dir / "DTOs"
    dtos_dir.mkdir(exist_ok=True)
    (dtos_dir / "AuthDTOs.cs").write_text(f"""namespace {nom}.DTOs;

public record RegisterRequest(string Username, string Email, string Password);
public record LoginRequest(string Email, string Password);
public record AuthResponse(string Token, string Username);
""")
    print("✅ DTOs créés : DTOs/AuthDTOs.cs")

    # === Créer le AuthController ===
    controllers_dir = projet_dir / "Controllers"
    controllers_dir.mkdir(exist_ok=True)
    (controllers_dir / "AuthController.cs").write_text(f"""using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.IdentityModel.Tokens;
using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Text;
using {nom}.Data;
using {nom}.DTOs;
using {nom}.Models;

namespace {nom}.Controllers;

[ApiController]
[Route("api/[controller]")]
public class AuthController : ControllerBase
{{
    private readonly {nom}DbContext _context;
    private readonly IConfiguration _config;

    public AuthController({nom}DbContext context, IConfiguration config)
    {{
        _context = context;
        _config = config;
    }}

    [HttpPost("register")]
    [Microsoft.AspNetCore.Authorization.AllowAnonymous]
    public async Task<IActionResult> Register(RegisterRequest request)
    {{
        if (await _context.Users.AnyAsync(u => u.Email == request.Email))
            return BadRequest("Un compte existe déjà avec cet email.");

        var user = new User
        {{
            Username = request.Username,
            Email = request.Email,
            PasswordHash = BCrypt.Net.BCrypt.HashPassword(request.Password)
        }};

        _context.Users.Add(user);
        await _context.SaveChangesAsync();

        return Ok(new {{ message = "Compte créé avec succès." }});
    }}

    [HttpPost("login")]
    [Microsoft.AspNetCore.Authorization.AllowAnonymous]
    public async Task<IActionResult> Login(LoginRequest request)
    {{
        var user = await _context.Users.FirstOrDefaultAsync(u => u.Email == request.Email);

        if (user is null || !BCrypt.Net.BCrypt.Verify(request.Password, user.PasswordHash))
            return Unauthorized("Email ou mot de passe incorrect.");

        var token = GenererToken(user);
        return Ok(new AuthResponse(token, user.Username));
    }}

    private string GenererToken(User user)
    {{
        var key = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(_config["Jwt:SecretKey"]!));
        var creds = new SigningCredentials(key, SecurityAlgorithms.HmacSha256);

        var claims = new[]
        {{
            new Claim(ClaimTypes.NameIdentifier, user.Id.ToString()),
            new Claim(ClaimTypes.Name, user.Username),
            new Claim(ClaimTypes.Email, user.Email)
        }};

        var token = new JwtSecurityToken(
            issuer: _config["Jwt:Issuer"],
            audience: _config["Jwt:Audience"],
            claims: claims,
            expires: DateTime.UtcNow.AddDays(7),
            signingCredentials: creds
        );

        return new JwtSecurityTokenHandler().WriteToken(token);
    }}
}}
""")
    print("✅ AuthController créé : Controllers/AuthController.cs")

# === 12. Build avec correction automatique par IA ===
def extraire_fichiers_erreurs(output):
    """Extrait les chemins de fichiers mentionnés dans les erreurs de build."""
    return list(set(re.findall(r'(/[\w/\-_.]+\.cs)', output)))

def corriger_avec_ollama(erreurs, fichier_path):
    """Demande à Ollama de corriger un fichier C# en fonction des erreurs."""
    try:
        contenu = Path(fichier_path).read_text()
        prompt = f"""Tu es un expert C# .NET 9. Voici des erreurs de build et le fichier concerné.
Retourne UNIQUEMENT le fichier corrigé, sans explication, sans balises markdown, juste le code C#.

Erreurs :
{erreurs}

Fichier ({fichier_path}) :
{contenu}"""

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral-nemo", "prompt": prompt, "stream": False},
            timeout=120
        )
        corrige = response.json().get("response", "").strip()
        # Retirer toutes les balises markdown (début et fin)
        corrige = re.sub(r'```[a-zA-Z]*', '', corrige)
        corrige = corrige.strip('`').strip()
        return corrige
    except Exception as e:
        print(f"  ⚠️  Ollama indisponible : {e}")
        return None

MAX_TENTATIVES = 3
print("⚙️  Vérification du build...")

for tentative in range(1, MAX_TENTATIVES + 1):
    build = subprocess.run("dotnet build --framework net9.0", shell=True, cwd=projet_dir, capture_output=True, text=True)

    if build.returncode == 0:
        print("✅ Build OK")
        break

    output_erreurs = build.stdout + build.stderr
    print(f"❌ Erreur de build (tentative {tentative}/{MAX_TENTATIVES})")

    if tentative == MAX_TENTATIVES:
        print(output_erreurs[-3000:])
        print("\n⚠️  Impossible de corriger automatiquement. Corrige manuellement et relance.")
        exit(1)

    # Trouver les fichiers concernés
    fichiers = extraire_fichiers_erreurs(output_erreurs)
    fichiers_existants = [f for f in fichiers if Path(f).exists()]

    if not fichiers_existants:
        print(output_erreurs[-2000:])
        exit(1)

    print(f"  🤖 Correction automatique via Ollama ({len(fichiers_existants)} fichier(s))...")
    for fichier in fichiers_existants:
        print(f"     → {Path(fichier).name}")
        corrige = corriger_avec_ollama(output_erreurs[-2000:], fichier)
        if corrige:
            Path(fichier).write_text(corrige)
            print(f"     ✅ Corrigé")

    print(f"  🔄 Nouveau build...")


# === 13. Créer la première migration ===
print("⚙️  Création de la migration initiale...")
run("dotnet ef migrations add InitialCreate", cwd=projet_dir)
print("✅ Migration 'InitialCreate' créée")

# === Résumé ===
auth_info = f"""
  🔑 JWT Secret : {jwt_secret}

Endpoints d'authentification :
  POST /api/auth/register  {{ "username": "", "email": "", "password": "" }}
  POST /api/auth/login     {{ "email": "", "password": "" }}
  → Retourne un token à mettre dans : Authorization: Bearer <token>
  → Protège tes routes avec [Authorize]
  → Swagger UI : http://localhost:5000/swagger""" if template in ("webapp", "webapi") else ""

print(f"""
{"=" * 55}
✅ Projet créé avec succès !
{"=" * 55}
  📁 Dossier    : {solution_dir}
  🏗️  Type       : {label_type} (.NET 9)
  🗄️  Base       : {db_type}
  📄 DbContext  : Data/{nom}DbContext.cs{auth_info}

Pour appliquer la migration :
  cd {solution_dir}/{nom} && dotnet ef database update

Pour lancer le projet :
  cd {solution_dir}/{nom} && dotnet run
{"=" * 55}
""")
