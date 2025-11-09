/* eslint-disable @angular-eslint/prefer-inject */
// src/app/features/landing/components/landing-page.component.ts

import { Component, HostListener, OnInit, inject } from '@angular/core';
import { Router } from '@angular/router';
import { CommonModule } from '@angular/common';
import { LucideAngularModule } from 'lucide-angular';
import { ZardAccordionComponent } from '@shared/components/accordion/accordion.component';
import { ZardAccordionItemComponent } from '@shared/components/accordion/accordion-item.component';


/**
 * Interfaz para características del producto.
 */
interface Feature {
  icon: any;
  title: string;
  description: string;
}

/**
 * Interfaz para perfiles de usuario objetivo.
 */
interface Persona {
  icon: any;
  title: string;
  description: string;
  benefits: string[];
}

/**
 * Interfaz para pasos de uso.
 */
interface Step {
  icon: any;
  title: string;
  description: string;
}

/**
 * Interfaz para planes de precios.
 */
interface PricingPlan {
  name: string;
  price: string;
  description: string;
  features: string[];
  cta: string;
  featured: boolean;
}

/**
 * Interfaz para testimonios.
 */
interface Testimonial {
  quote: string;
  author: string;
  position: string;
}

/**
 * Landing Page principal de la aplicación.
 * Presenta características, casos de uso, pasos, precios y testimonios.
 */
@Component({
  selector: 'app-landing-page',
  standalone: true,
  imports: [
    CommonModule,
    LucideAngularModule,
    ZardAccordionComponent,
    ZardAccordionItemComponent
  ],
  templateUrl: './landing-page.html',
  styleUrl: './landing-page.css'
})
export class LandingPage implements OnInit {
  // ==================================================================
  // SERVICIOS
  // ==================================================================

  private router = inject(Router);

  // ==================================================================
  // DATOS ESTÁTICOS
  // ==================================================================

  readonly features: Feature[] = [
    {
      icon: 'fas fa-cloud-upload-alt',
      title: 'Carga Instantánea',
      description: 'Sube archivos en formato PDF, DOCX, y TXT con solo arrastrar y soltar. Todo se almacena de forma segura en la nube.'
    },
    {
      icon: 'fas fa-search',
      title: 'Búsqueda Inteligente',
      description: 'Encuentra información específica dentro de tus documentos con búsquedas semánticas. No solo por palabras clave, sino por significado.'
    },
    {
      icon: 'fas fa-microphone',
      title: 'Asistente por Voz',
      description: 'Interactúa con tus documentos de forma natural. Pregúntale a tu asistente por voz y obtén respuestas precisas al instante.'
    },
    {
      icon: 'fas fa-history',
      title: 'Historial Organizado',
      description: 'Accede a un historial completo de todas tus interacciones y búsquedas. Nunca pierdas el hilo de tus consultas.'
    },
    {
      icon: 'fas fa-file-contract',
      title: 'Resúmenes Automáticos',
      description: 'Pide a la IA que genere resúmenes de documentos extensos. Extrae los puntos clave y ahorra horas de lectura.'
    },
    {
      icon: 'fas fa-lock',
      title: 'Seguridad Robusta',
      description: 'Tus documentos y datos están protegidos con cifrado de grado militar. Tu privacidad es nuestra máxima prioridad.'
    }
  ];

  readonly personas: Persona[] = [
    {
      icon: 'Scale',
      title: 'Abogados y Profesionales Legales',
      description: 'Gestiona contratos, casos legales y documentación jurídica de manera eficiente.',
      benefits: [
        'Búsqueda rápida de cláusulas y precedentes',
        'Comparación automática de documentos',
        'Resúmenes de casos complejos'
      ]
    },
    {
      icon: 'chart-bar',
      title: 'Analistas e Investigadores',
      description: 'Procesa grandes volúmenes de datos y extrae insights valiosos en minutos.',
      benefits: [
        'Análisis de reportes extensos',
        'Extracción de datos clave',
        'Generación de informes sintéticos'
      ]
    },
    {
      icon: 'graduation-cap',
      title: 'Estudiantes y Académicos',
      description: 'Estudia de manera más inteligente con resúmenes y búsquedas avanzadas.',
      benefits: [
        'Resúmenes de papers y libros',
        'Búsqueda de conceptos específicos',
        'Organización de referencias'
      ]
    },
    {
      icon: 'Briefcase',
      title: 'Ejecutivos y Gerentes',
      description: 'Toma decisiones informadas basadas en análisis rápido de documentos.',
      benefits: [
        'Revisión express de propuestas',
        'Extracción de métricas clave',
        'Síntesis de reportes financieros'
      ]
    },
    {
      icon: 'Stethoscope',
      title: 'Profesionales de la Salud',
      description: 'Accede rápidamente a información médica y expedientes de pacientes.',
      benefits: [
        'Búsqueda de historial clínico',
        'Resúmenes de estudios médicos',
        'Gestión de documentación sanitaria'
      ]
    },
    {
      icon: 'Building',
      title: 'Equipos Empresariales',
      description: 'Colabora eficientemente con acceso compartido a documentos centralizados.',
      benefits: [
        'Gestión de conocimiento corporativo',
        'Onboarding acelerado',
        'Base de datos compartida'
      ]
    }
  ];

  readonly steps: Step[] = [
    {
      icon: 'folder-open',
      title: 'Sube tus Documentos',
      description: 'Arrastra y suelta tus archivos PDF, DOCX o TXT. También puedes subirlos desde tu almacenamiento en la nube.'
    },
    {
      icon: 'Bot',
      title: 'Haz tus Preguntas',
      description: 'Escribe o usa tu voz para hacer preguntas sobre tus documentos. La IA comprende el contexto y el lenguaje natural.'
    },
    {
      icon: 'Sparkles',
      title: 'Obtén Respuestas Instantáneas',
      description: 'Recibe respuestas precisas, resúmenes detallados y referencias exactas en segundos, no en horas.'
    }
  ];

  readonly pricingPlans: PricingPlan[] = [
    {
      name: 'Básico',
      price: '9',
      description: 'Perfecto para uso personal',
      features: [
        'Hasta 50 documentos',
        '100 consultas/mes',
        'Asistente por voz',
        'Resúmenes automáticos',
        'Soporte por email'
      ],
      cta: 'Comenzar Gratis',
      featured: false
    },
    {
      name: 'Pro',
      price: '29',
      description: 'Para profesionales exigentes',
      features: [
        'Documentos ilimitados',
        'Consultas ilimitadas',
        'Asistente por voz avanzado',
        'Búsqueda semántica',
        'Exportación de resultados',
        'Soporte prioritario',
        'API de integración'
      ],
      cta: 'Empezar Ahora',
      featured: true
    },
    {
      name: 'Enterprise',
      price: '99',
      description: 'Para equipos y empresas',
      features: [
        'Todo en Pro +',
        'Usuarios ilimitados',
        'Almacenamiento ilimitado',
        'Integraciones personalizadas',
        'Soporte 24/7',
        'Gerente de cuenta dedicado',
        'SLA garantizado',
        'Auditoría y compliance'
      ],
      cta: 'Contactar Ventas',
      featured: false
    }
  ];

  readonly testimonials: Testimonial[] = [
    {
      quote: 'Esta herramienta ha cambiado mi forma de gestionar la información. Encontrar datos en mis informes nunca había sido tan fácil. Es como tener un asistente personal 24/7.',
      author: 'Pedro G.',
      position: 'Gerente de Proyectos'
    },
    {
      quote: 'El asistente por voz es increíblemente preciso. Le pregunto por un dato en un documento legal y me lo da en segundos. El tiempo que he ahorrado es invaluable.',
      author: 'Laura M.',
      position: 'Abogada'
    },
    {
      quote: 'La función de resumen es una maravilla. Me permite ponerme al día con documentos largos de manera rápida y eficiente. Muy recomendable para cualquier estudiante o profesional.',
      author: 'Ana R.',
      position: 'Analista de Datos'
    }
  ];

  // ==================================================================
  // ANIMACIONES DE SCROLL
  // ==================================================================

  private intersectionObserver?: IntersectionObserver;

  ngOnInit(): void {
    this.setupScrollAnimations();
  }

  /** Configura animaciones de aparición al hacer scroll */
  private setupScrollAnimations(): void {
    const options: IntersectionObserverInit = {
      threshold: 0.1,
      rootMargin: '0px 0px -100px 0px'
    };

    this.intersectionObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
        }
      });
    }, options);

    // Delay para asegurar que el DOM esté listo
    setTimeout(() => {
      document.querySelectorAll('.animate-on-scroll').forEach(el => {
        this.intersectionObserver?.observe(el);
      });
    }, 100);
  }

  @HostListener('window:scroll')
  onWindowScroll(): void {
    this.animateOnScroll();
  }

  /** Animación fallback (en caso de que IntersectionObserver no funcione) */
  private animateOnScroll(): void {
    document.querySelectorAll('.animate-on-scroll').forEach(el => {
      const rect = el.getBoundingClientRect();
      if (rect.top < window.innerHeight - 100) {
        el.classList.add('visible');
      }
    });
  }

  // ==================================================================
  // NAVEGACIÓN
  // ==================================================================

  /** Redirige al login */
  goToLogin(): void {
    this.router.navigate(['/login']);
  }

  mobileMenuOpen = false;

  toggleMobileMenu(): void {
    this.mobileMenuOpen = !this.mobileMenuOpen;
  }

  closeMobileMenu(): void {
    this.mobileMenuOpen = false;
  }

}