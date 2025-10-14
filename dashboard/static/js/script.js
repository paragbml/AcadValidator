/**
 * Government Portal JavaScript - UGC-AICTE Performance Portal
 * Interactive functionality for the institutional performance tracker
 */

// Global Configuration
const CONFIG = {
    API_BASE_URL: window.location.origin,
    CHART_COLORS: {
        primary: '#002D62',
        secondary: '#0088cc',
        success: '#28a745',
        warning: '#ffc107',
        danger: '#dc3545',
        info: '#17a2b8',
        light: '#f8f9fa',
        dark: '#343a40'
    },
    PERFORMANCE_THRESHOLDS: {
        excellent: 85,
        veryGood: 75,
        good: 65,
        average: 55
    }
};

// Utility Functions
const Utils = {
    /**
     * Format number with commas
     */
    formatNumber: (num) => {
        return new Intl.NumberFormat('en-IN').format(num);
    },

    /**
     * Get performance level based on score
     */
    getPerformanceLevel: (score) => {
        if (score >= CONFIG.PERFORMANCE_THRESHOLDS.excellent) return 'Excellent';
        if (score >= CONFIG.PERFORMANCE_THRESHOLDS.veryGood) return 'Very Good';
        if (score >= CONFIG.PERFORMANCE_THRESHOLDS.good) return 'Good';
        if (score >= CONFIG.PERFORMANCE_THRESHOLDS.average) return 'Average';
        return 'Needs Improvement';
    },

    /**
     * Get color based on performance score
     */
    getPerformanceColor: (score) => {
        if (score >= CONFIG.PERFORMANCE_THRESHOLDS.excellent) return CONFIG.CHART_COLORS.success;
        if (score >= CONFIG.PERFORMANCE_THRESHOLDS.veryGood) return CONFIG.CHART_COLORS.info;
        if (score >= CONFIG.PERFORMANCE_THRESHOLDS.good) return CONFIG.CHART_COLORS.primary;
        if (score >= CONFIG.PERFORMANCE_THRESHOLDS.average) return CONFIG.CHART_COLORS.warning;
        return CONFIG.CHART_COLORS.danger;
    },

    /**
     * Show loading state
     */
    showLoading: (elementId) => {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = `
                <div class="text-center p-4">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p class="mt-2">Processing...</p>
                </div>
            `;
        }
    },

    /**
     * Show error message
     */
    showError: (elementId, message) => {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = `
                <div class="alert alert-danger">
                    <i class="bi bi-exclamation-triangle"></i>
                    ${message}
                </div>
            `;
        }
    },

    /**
     * Debounce function
     */
    debounce: (func, wait) => {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    /**
     * Animate number counting
     */
    animateNumber: (element, start, end, duration = 1000) => {
        const range = end - start;
        const increment = range / (duration / 16);
        let current = start;
        
        const timer = setInterval(() => {
            current += increment;
            if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
                current = end;
                clearInterval(timer);
            }
            element.textContent = Math.round(current * 10) / 10;
        }, 16);
    }
};

// Clock and Time Display
const TimeDisplay = {
    init: () => {
        TimeDisplay.updateClock();
        setInterval(TimeDisplay.updateClock, 1000);
        TimeDisplay.updateLastUpdated();
    },

    updateClock: () => {
        const now = new Date();
        const timeString = now.toLocaleTimeString('en-IN', {
            hour12: true,
            hour: 'numeric',
            minute: '2-digit'
        });
        
        const timeElement = document.getElementById('current-time');
        if (timeElement) {
            timeElement.textContent = timeString;
        }
    },

    updateLastUpdated: () => {
        const lastUpdatedElement = document.getElementById('last-updated');
        if (lastUpdatedElement) {
            const now = new Date();
            lastUpdatedElement.textContent = now.toLocaleDateString('en-IN', {
                year: 'numeric',
                month: 'short',
                day: 'numeric'
            });
        }
    }
};

// Range Slider Synchronization
const RangeSliders = {
    init: () => {
        const sliderPairs = [
            { input: 'aicte_score', range: 'aicte_range' },
            { input: 'ugc_rating', range: 'ugc_range' },
            { input: 'nirf_rank', range: 'nirf_range' },
            { input: 'placement_percentage', range: 'placement_range' },
            { input: 'faculty_ratio', range: 'faculty_range' },
            { input: 'research_projects', range: 'research_range' },
            { input: 'infrastructure_score', range: 'infrastructure_range' },
            { input: 'satisfaction_score', range: 'satisfaction_range' }
        ];

        sliderPairs.forEach(pair => {
            RangeSliders.syncInputs(pair.input, pair.range);
        });
    },

    syncInputs: (inputId, rangeId) => {
        const input = document.getElementById(inputId);
        const range = document.getElementById(rangeId);

        if (input && range) {
            // Sync range to input
            input.addEventListener('input', () => {
                range.value = input.value;
                RangeSliders.updateVisualFeedback(inputId, input.value);
            });

            // Sync input to range
            range.addEventListener('input', () => {
                input.value = range.value;
                RangeSliders.updateVisualFeedback(inputId, range.value);
            });

            // Initial visual feedback
            RangeSliders.updateVisualFeedback(inputId, input.value);
        }
    },

    updateVisualFeedback: (inputId, value) => {
        const input = document.getElementById(inputId);
        if (!input) return;

        // Add visual feedback based on value
        input.classList.remove('is-valid', 'is-invalid');
        
        if (inputId.includes('score') || inputId.includes('percentage')) {
            if (parseFloat(value) >= 80) {
                input.classList.add('is-valid');
            } else if (parseFloat(value) < 50) {
                input.classList.add('is-invalid');
            }
        }
    }
};

// Form Validation and Submission
const FormHandler = {
    init: () => {
        const form = document.getElementById('predictionForm');
        if (form) {
            form.addEventListener('submit', FormHandler.handleSubmit);
        }
    },

    handleSubmit: async (event) => {
        event.preventDefault();
        
        const formData = FormHandler.collectFormData();
        if (!FormHandler.validateFormData(formData)) {
            return;
        }

        FormHandler.showLoadingState();
        
        try {
            const result = await FormHandler.submitPrediction(formData);
            FormHandler.displayResults(result);
        } catch (error) {
            FormHandler.showError('Failed to get prediction. Please try again.');
            console.error('Prediction error:', error);
        }
    },

    collectFormData: () => {
        return {
            AICTE_Approval_Score: parseFloat(document.getElementById('aicte_score').value),
            UGC_Rating: parseFloat(document.getElementById('ugc_rating').value),
            NIRF_Rank: parseInt(document.getElementById('nirf_rank').value),
            Placement_Percentage: parseFloat(document.getElementById('placement_percentage').value),
            Faculty_Student_Ratio: parseFloat(document.getElementById('faculty_ratio').value),
            Research_Projects: parseInt(document.getElementById('research_projects').value),
            Infrastructure_Score: parseFloat(document.getElementById('infrastructure_score').value),
            Student_Satisfaction_Score: parseFloat(document.getElementById('satisfaction_score').value)
        };
    },

    validateFormData: (data) => {
        const errors = [];

        // Validation rules
        const validations = [
            { field: 'AICTE_Approval_Score', min: 40, max: 100, name: 'AICTE Approval Score' },
            { field: 'UGC_Rating', min: 3, max: 10, name: 'UGC Rating' },
            { field: 'NIRF_Rank', min: 1, max: 200, name: 'NIRF Rank' },
            { field: 'Placement_Percentage', min: 25, max: 100, name: 'Placement Percentage' },
            { field: 'Faculty_Student_Ratio', min: 8, max: 25, name: 'Faculty-Student Ratio' },
            { field: 'Research_Projects', min: 0, max: 50, name: 'Research Projects' },
            { field: 'Infrastructure_Score', min: 40, max: 100, name: 'Infrastructure Score' },
            { field: 'Student_Satisfaction_Score', min: 40, max: 100, name: 'Student Satisfaction Score' }
        ];

        validations.forEach(rule => {
            const value = data[rule.field];
            if (isNaN(value) || value < rule.min || value > rule.max) {
                errors.push(`${rule.name} must be between ${rule.min} and ${rule.max}`);
            }
        });

        if (errors.length > 0) {
            FormHandler.showValidationErrors(errors);
            return false;
        }

        return true;
    },

    showValidationErrors: (errors) => {
        const errorHtml = `
            <div class="alert alert-danger">
                <h6><i class="bi bi-exclamation-triangle"></i> Please correct the following errors:</h6>
                <ul class="mb-0">
                    ${errors.map(error => `<li>${error}</li>`).join('')}
                </ul>
            </div>
        `;
        
        const resultsContainer = document.getElementById('prediction-results');
        if (resultsContainer) {
            resultsContainer.innerHTML = errorHtml;
        }
    },

    showLoadingState: () => {
        const resultsContainer = document.getElementById('prediction-results');
        const loadingSpinner = document.getElementById('loading-spinner');
        
        if (resultsContainer) {
            resultsContainer.style.display = 'none';
        }
        if (loadingSpinner) {
            loadingSpinner.classList.remove('d-none');
        }
    },

    submitPrediction: async (formData) => {
        const response = await fetch(`${CONFIG.API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    },

    displayResults: (result) => {
        const resultsContainer = document.getElementById('prediction-results');
        const loadingSpinner = document.getElementById('loading-spinner');
        
        // Hide loading spinner
        if (loadingSpinner) {
            loadingSpinner.classList.add('d-none');
        }

        if (!result.success) {
            FormHandler.showError(result.error || 'Prediction failed');
            return;
        }

        // Display results
        const prediction = result.prediction;
        const recommendations = result.recommendations || [];

        const resultsHtml = `
            <div class="prediction-success">
                <div class="text-center mb-4">
                    <div class="performance-score-display">
                        <div class="score-circle" style="background: conic-gradient(${Utils.getPerformanceColor(prediction.performance_index)} 0deg, ${Utils.getPerformanceColor(prediction.performance_index)} ${prediction.performance_index * 3.6}deg, #e9ecef ${prediction.performance_index * 3.6}deg);">
                            <div class="score-inner">
                                <span class="score-number">${prediction.performance_index}</span>
                                <span class="score-suffix">%</span>
                            </div>
                        </div>
                    </div>
                    <h4 class="mt-3">${prediction.performance_level}</h4>
                    <p class="text-muted">${prediction.level_description}</p>
                    <span class="badge bg-${prediction.level_color} fs-6">${prediction.performance_level}</span>
                </div>

                <div class="recommendations-section">
                    <h6><i class="bi bi-lightbulb text-warning"></i> Key Recommendations</h6>
                    <div class="recommendations-list">
                        ${recommendations.slice(0, 4).map(rec => `
                            <div class="recommendation-item">
                                <i class="bi bi-check-circle text-success"></i>
                                <span>${rec}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>

                <div class="mt-4 text-center">
                    <button class="btn btn-primary" onclick="viewDetailedResults()">
                        <i class="bi bi-file-text"></i> View Detailed Report
                    </button>
                    <button class="btn btn-outline-secondary" onclick="resetForm()">
                        <i class="bi bi-arrow-clockwise"></i> New Prediction
                    </button>
                </div>
            </div>
        `;

        if (resultsContainer) {
            resultsContainer.innerHTML = resultsHtml;
            resultsContainer.style.display = 'block';
            
            // Animate score counting
            const scoreElement = resultsContainer.querySelector('.score-number');
            if (scoreElement) {
                Utils.animateNumber(scoreElement, 0, prediction.performance_index, 1500);
            }
        }

        // Store results for detailed view
        window.lastPredictionResult = result;
    },

    showError: (message) => {
        Utils.showError('prediction-results', message);
        
        const loadingSpinner = document.getElementById('loading-spinner');
        if (loadingSpinner) {
            loadingSpinner.classList.add('d-none');
        }
        
        const resultsContainer = document.getElementById('prediction-results');
        if (resultsContainer) {
            resultsContainer.style.display = 'block';
        }
    }
};

// Sample Data Loading
const SampleData = {
    tier1: {
        AICTE_Approval_Score: 92.5,
        UGC_Rating: 8.8,
        NIRF_Rank: 15,
        Placement_Percentage: 89.5,
        Faculty_Student_Ratio: 9.2,
        Research_Projects: 35,
        Infrastructure_Score: 94.0,
        Student_Satisfaction_Score: 88.5
    },
    tier2: {
        AICTE_Approval_Score: 78.5,
        UGC_Rating: 7.2,
        NIRF_Rank: 65,
        Placement_Percentage: 72.5,
        Faculty_Student_Ratio: 12.8,
        Research_Projects: 18,
        Infrastructure_Score: 80.0,
        Student_Satisfaction_Score: 75.5
    },
    tier3: {
        AICTE_Approval_Score: 65.0,
        UGC_Rating: 5.8,
        NIRF_Rank: 125,
        Placement_Percentage: 58.5,
        Faculty_Student_Ratio: 18.5,
        Research_Projects: 8,
        Infrastructure_Score: 68.0,
        Student_Satisfaction_Score: 62.5
    }
};

// Global Functions (called from HTML)
window.setupRangeSliders = () => {
    RangeSliders.init();
};

window.setupFormValidation = () => {
    FormHandler.init();
};

window.loadSampleData = (tier) => {
    const data = SampleData[tier];
    if (!data) return;

    // Load data into form fields
    Object.keys(data).forEach(key => {
        const fieldMap = {
            AICTE_Approval_Score: 'aicte_score',
            UGC_Rating: 'ugc_rating',
            NIRF_Rank: 'nirf_rank',
            Placement_Percentage: 'placement_percentage',
            Faculty_Student_Ratio: 'faculty_ratio',
            Research_Projects: 'research_projects',
            Infrastructure_Score: 'infrastructure_score',
            Student_Satisfaction_Score: 'satisfaction_score'
        };

        const fieldId = fieldMap[key];
        const input = document.getElementById(fieldId);
        const range = document.getElementById(fieldId.replace('_', '_range'));

        if (input) {
            input.value = data[key];
            if (range) {
                range.value = data[key];
            }
        }
    });

    // Close modal
    const modal = bootstrap.Modal.getInstance(document.getElementById('sampleDataModal'));
    if (modal) {
        modal.hide();
    }

    // Show success message
    const toast = `
        <div class="toast align-items-center text-white bg-success border-0" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="d-flex">
                <div class="toast-body">
                    <i class="bi bi-check-circle"></i> Sample data loaded for ${tier.charAt(0).toUpperCase() + tier.slice(1)} institution
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        </div>
    `;
    
    showToast(toast);
};

window.resetForm = () => {
    const form = document.getElementById('predictionForm');
    if (form) {
        form.reset();
        
        // Reset range sliders
        const ranges = form.querySelectorAll('input[type="range"]');
        ranges.forEach(range => {
            const input = document.getElementById(range.id.replace('_range', ''));
            if (input) {
                range.value = input.value;
            }
        });

        // Clear results
        const resultsContainer = document.getElementById('prediction-results');
        if (resultsContainer) {
            resultsContainer.innerHTML = `
                <div class="placeholder-content">
                    <i class="bi bi-graph-up display-1 text-muted"></i>
                    <h4 class="text-muted mt-3">Performance Analysis</h4>
                    <p class="text-muted">Enter institutional metrics and click "Predict Performance" to see AI-powered analysis</p>
                </div>
            `;
        }
    }
};

window.viewDetailedResults = () => {
    if (window.lastPredictionResult) {
        // Create a form and submit to results page
        const form = document.createElement('form');
        form.method = 'POST';
        form.action = '/results';
        
        const dataInput = document.createElement('input');
        dataInput.type = 'hidden';
        dataInput.name = 'prediction_data';
        dataInput.value = JSON.stringify(window.lastPredictionResult);
        
        form.appendChild(dataInput);
        document.body.appendChild(form);
        form.submit();
    }
};

window.exportResults = () => {
    if (window.lastPredictionResult) {
        // Implement PDF export
        window.print();
    } else {
        showToast(`
            <div class="toast align-items-center text-white bg-warning border-0" role="alert">
                <div class="d-flex">
                    <div class="toast-body">
                        <i class="bi bi-exclamation-triangle"></i> No results available to export. Please run a prediction first.
                    </div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                </div>
            </div>
        `);
    }
};

// Toast notification system
function showToast(toastHtml) {
    // Create toast container if it doesn't exist
    let toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toast-container';
        toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
        toastContainer.style.zIndex = '1056';
        document.body.appendChild(toastContainer);
    }

    // Add toast to container
    const toastWrapper = document.createElement('div');
    toastWrapper.innerHTML = toastHtml;
    toastContainer.appendChild(toastWrapper.firstElementChild);

    // Initialize and show toast
    const toastElement = toastContainer.lastElementChild;
    const toast = new bootstrap.Toast(toastElement, { delay: 4000 });
    toast.show();

    // Remove toast after it's hidden
    toastElement.addEventListener('hidden.bs.toast', () => {
        toastElement.remove();
    });
}

// Keyboard Shortcuts
document.addEventListener('keydown', (event) => {
    // Ctrl + Enter to submit prediction form
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        const form = document.getElementById('predictionForm');
        if (form && document.activeElement && form.contains(document.activeElement)) {
            event.preventDefault();
            form.dispatchEvent(new Event('submit'));
        }
    }
    
    // Esc to close modals
    if (event.key === 'Escape') {
        const openModals = document.querySelectorAll('.modal.show');
        openModals.forEach(modal => {
            const modalInstance = bootstrap.Modal.getInstance(modal);
            if (modalInstance) {
                modalInstance.hide();
            }
        });
    }
});

// Accessibility Improvements
document.addEventListener('DOMContentLoaded', () => {
    // Add skip link
    const skipLink = document.createElement('a');
    skipLink.href = '#main-content';
    skipLink.className = 'visually-hidden-focusable btn btn-primary position-absolute top-0 start-0 m-2';
    skipLink.style.zIndex = '1057';
    skipLink.textContent = 'Skip to main content';
    document.body.insertBefore(skipLink, document.body.firstChild);

    // Add main content id
    const mainContent = document.querySelector('.main-content');
    if (mainContent) {
        mainContent.id = 'main-content';
        mainContent.setAttribute('tabindex', '-1');
    }

    // Announce page changes for screen readers
    const announcer = document.createElement('div');
    announcer.setAttribute('aria-live', 'polite');
    announcer.setAttribute('aria-atomic', 'true');
    announcer.className = 'visually-hidden';
    announcer.id = 'page-announcer';
    document.body.appendChild(announcer);
});

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    TimeDisplay.init();
    RangeSliders.init();
    FormHandler.init();
    
    // Add smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
});

// Service Worker Registration (for PWA capabilities)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then((registration) => {
                console.log('SW registered: ', registration);
            })
            .catch((registrationError) => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}

// Performance monitoring
window.addEventListener('load', () => {
    // Log page load time
    const loadTime = performance.timing.loadEventEnd - performance.timing.navigationStart;
    console.log(`Page loaded in ${loadTime}ms`);
    
    // Monitor for layout shifts
    if ('LayoutShift' in window) {
        let cls = 0;
        new PerformanceObserver((entryList) => {
            for (const entry of entryList.getEntries()) {
                if (!entry.hadRecentInput) {
                    cls += entry.value;
                }
            }
        }).observe({type: 'layout-shift', buffered: true});
    }
});

// Navigation Functions for Enhanced Menu System
const NavigationHandler = {
    /**
     * Show Comparative Analysis section
     */
    showComparativeAnalysis: () => {
        showToast('Loading Comparative Analysis...', 'info');
        // Implement comparative analysis functionality
        setTimeout(() => {
            showToast('Comparative Analysis feature coming soon!', 'warning');
        }, 1000);
    },

    /**
     * Show Trend Insights
     */
    showTrendInsights: () => {
        showToast('Loading Trend Insights...', 'info');
        // Implement trend insights functionality
        setTimeout(() => {
            showToast('Trend Insights feature coming soon!', 'warning');
        }, 1000);
    },

    /**
     * Show Dataset Explorer
     */
    showDatasetExplorer: () => {
        // Navigate to reports page which has dataset functionality
        window.location.href = '/reports';
    },

    /**
     * Show Data Upload interface
     */
    showDataUpload: () => {
        showToast('Data Upload feature coming soon!', 'info');
        // In future versions, this would open a modal for CSV/Excel upload
    },

    /**
     * Show Data Cleaning Assistant
     */
    showDataCleaning: () => {
        showToast('Data Quality Assistant coming soon!', 'info');
        // Future: AI-powered data cleaning suggestions
    },

    /**
     * Show Blockchain Verification
     */
    showBlockchainVerification: () => {
        showToast('Blockchain Verification available in prediction results!', 'info');
        // Navigate to prediction section
        window.location.href = '/#prediction';
    },

    /**
     * Show Compliance Tracker
     */
    showComplianceTracker: () => {
        showToast('Compliance Tracker feature coming soon!', 'info');
        // Future: UGC/AICTE compliance monitoring
    },

    /**
     * Show Digital Participation Index
     */
    showDigitalParticipation: () => {
        showToast('Digital Participation Index coming soon!', 'info');
        // Future: SWAYAM, NPTEL participation tracking
    },

    /**
     * Show AI Recommendation Engine
     */
    showRecommendationEngine: () => {
        showToast('AI Recommendations available in prediction results!', 'info');
        // Navigate to prediction section
        window.location.href = '/#prediction';
    },

    /**
     * Show Institution Trust Index
     */
    showTrustIndex: () => {
        showToast('Institution Trust Index feature coming soon!', 'info');
        // Future: Trust scoring based on multiple factors
    },

    /**
     * Show Risk Heatmap
     */
    showRiskHeatmap: () => {
        showToast('Risk Heatmap feature coming soon!', 'info');
        // Future: Red-Amber-Green risk visualization
    },

    /**
     * Generate AI Report
     */
    generateAIReport: () => {
        showToast('Generating AI Report...', 'info');
        // Simulate report generation
        setTimeout(() => {
            showToast('AI Report generated! Check your downloads.', 'success');
            // In production, this would call the backend API
        }, 2000);
    },

    /**
     * Download Institution Summary
     */
    downloadInstitutionSummary: () => {
        // Navigate to reports page for download functionality
        window.location.href = '/reports';
    },

    /**
     * Export Performance Data
     */
    exportPerformanceData: () => {
        showToast('Exporting Performance Data...', 'info');
        // Future: CSV/Excel export functionality
        setTimeout(() => {
            showToast('Export feature coming soon!', 'warning');
        }, 1000);
    },

    /**
     * Generate Automated Feedback
     */
    generateAutomatedFeedback: () => {
        showToast('Generating Automated Feedback...', 'info');
        // Future: AI-powered feedback generation
        setTimeout(() => {
            showToast('Automated Feedback feature coming soon!', 'warning');
        }, 1000);
    }
};

// Make navigation functions globally available
window.showComparativeAnalysis = NavigationHandler.showComparativeAnalysis;
window.showTrendInsights = NavigationHandler.showTrendInsights;
window.showDatasetExplorer = NavigationHandler.showDatasetExplorer;
window.showDataUpload = NavigationHandler.showDataUpload;
window.showDataCleaning = NavigationHandler.showDataCleaning;
window.showBlockchainVerification = NavigationHandler.showBlockchainVerification;
window.showComplianceTracker = NavigationHandler.showComplianceTracker;
window.showDigitalParticipation = NavigationHandler.showDigitalParticipation;
window.showRecommendationEngine = NavigationHandler.showRecommendationEngine;
window.showTrustIndex = NavigationHandler.showTrustIndex;
window.showRiskHeatmap = NavigationHandler.showRiskHeatmap;
window.generateAIReport = NavigationHandler.generateAIReport;
window.downloadInstitutionSummary = NavigationHandler.downloadInstitutionSummary;
window.exportPerformanceData = NavigationHandler.exportPerformanceData;
window.generateAutomatedFeedback = NavigationHandler.generateAutomatedFeedback;

// Export for use in other modules
window.AcadValidatorApp = {
    Utils,
    CONFIG,
    FormHandler,
    RangeSliders,
    TimeDisplay,
    SampleData,
    NavigationHandler,
    showToast
};